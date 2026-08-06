"""Structured logging with request correlation.

Every log line carries the ``request_id`` of the request that produced it,
propagated through a :class:`~contextvars.ContextVar`. That is what makes a
production incident tractable: given one failing request id from a client, you
can retrieve every line the service emitted while handling it, including lines
from deep inside the batching engine that have no access to the request object.

``contextvars`` is the right mechanism because asyncio tasks inherit context on
creation, so the id follows the request across ``await`` boundaries without
being threaded through every function signature.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from collections.abc import MutableMapping
from contextvars import ContextVar
from typing import Any

from .config import LoggingConfig

# Set by the request-id middleware; read by the log formatter.
request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Attributes the stdlib puts on every LogRecord. Passing any of these via
# ``extra=`` makes ``Logger.makeRecord`` raise
#     KeyError: Attempt to overwrite 'name' in LogRecord
# which turns a harmless log line into a crash. ``extra={"name": ...}`` is an
# extremely easy mistake to make, so the adapter below renames collisions
# instead of letting them propagate.
_RESERVED = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
        "color_message",
    }
)


def new_request_id() -> str:
    return uuid.uuid4().hex


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with the request id injected."""

    def __init__(self, *, service: str = "inference-service", environment: str = "development"):
        super().__init__()
        self._service = service
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "service": self._service,
            "env": self._environment,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if request_id := request_id_var.get(""):
            payload["request_id"] = request_id

        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            payload[key] = _coerce(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = self.formatStack(record.stack_info)
        return json.dumps(payload, default=str, separators=(",", ":"))


class TextFormatter(logging.Formatter):
    """Human-readable single line, for local development."""

    def __init__(self) -> None:
        super().__init__(fmt="%(asctime)s %(levelname)-7s %(name)-28s %(message)s")
        self.default_time_format = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        if request_id := request_id_var.get(""):
            base = f"{base} [rid={request_id[:8]}]"
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED and not key.startswith("_")
        }
        if extras:
            base += " " + " ".join(f"{k}={_coerce(v)}" for k, v in sorted(extras.items()))
        return base


def _coerce(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_coerce(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce(v) for k, v in value.items()}
    return str(value)


def configure_logging(
    config: LoggingConfig,
    *,
    service_name: str = "inference-service",
    environment: str = "development",
) -> None:
    """Install handlers on the package and uvicorn loggers. Idempotent."""
    formatter: logging.Formatter = (
        JsonFormatter(service=service_name, environment=environment)
        if config.format == "json"
        else TextFormatter()
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    level = getattr(logging, config.level.upper(), logging.INFO)

    root = logging.getLogger("inference_service")
    root.setLevel(level)
    root.propagate = False
    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()
    root.addHandler(handler)

    # Uvicorn installs its own colourised handlers. Left alone, every request
    # produces one structured line from us and one unstructured line from
    # uvicorn, and the log backend can only parse half of them.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
        uvicorn_logger.setLevel(level)

    # We emit our own access log with timing, route template and request id.
    # Uvicorn's adds nothing and doubles the volume.
    logging.getLogger("uvicorn.access").disabled = True


class _SafeExtraAdapter(logging.LoggerAdapter[logging.Logger]):
    """Renames ``extra`` keys that would collide with LogRecord attributes.

    Without this, ``logger.info("loaded", extra={"name": "sentiment"})`` raises
    ``KeyError: Attempt to overwrite 'name' in LogRecord`` — from inside the
    logging call, at runtime, in whatever code path happened to log. That is an
    absurd way to take down a request, and the failure mode is easy to
    reintroduce months later.

    Colliding keys are prefixed with ``ctx_`` rather than dropped, so the
    context still reaches the log backend and the collision is visible in the
    output instead of silently swallowed.
    """

    def process(
        self, msg: Any, kwargs: MutableMapping[str, Any]
    ) -> tuple[Any, MutableMapping[str, Any]]:
        extra = kwargs.get("extra")
        if extra:
            kwargs["extra"] = {
                (f"ctx_{key}" if key in _RESERVED else key): value for key, value in extra.items()
            }
        return msg, kwargs


def get_logger(name: str) -> _SafeExtraAdapter:
    """Return a logger that cannot be crashed by a colliding ``extra`` key."""
    return _SafeExtraAdapter(logging.getLogger(f"inference_service.{name}"), {})
