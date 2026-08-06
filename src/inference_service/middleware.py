"""HTTP middleware: request correlation, access logging, metrics.

One middleware does all three because they need the same timing boundary, and
because each additional middleware layer adds a measurable per-request cost in
Starlette's ASGI chain.

**Route template, not raw path.** Metrics label on ``request.scope["route"]``'s
path format (``/v1/predict``), never ``request.url.path``. If a route ever gains
a path parameter, labelling on the raw path would create one Prometheus time
series per distinct value — the classic cardinality explosion that takes down a
monitoring stack.
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response
from starlette.types import ASGIApp

from .config import Config
from .logging_setup import get_logger, new_request_id, request_id_var
from .metrics import ServiceMetrics

logger = get_logger("http")

REQUEST_ID_HEADER = "X-Request-ID"

# Paths excluded from access logging. Health probes fire every few seconds from
# the orchestrator and drown out real traffic in the log backend; their outcome
# is already visible in metrics.
_QUIET_PATHS = frozenset({"/health", "/healthz", "/live", "/ready", "/readyz", "/metrics"})


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """Assigns a request id, records metrics, and emits one access log line."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        config: Config,
        metrics: ServiceMetrics | None = None,
    ) -> None:
        super().__init__(app)
        self._config = config
        self._metrics = metrics
        self._access_log = config.logging.access_log

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Honour a caller-supplied id so a trace spans the whole call chain
        # through an API gateway; mint one otherwise.
        incoming = request.headers.get(REQUEST_ID_HEADER, "").strip()
        request_id = incoming[:64] if incoming else new_request_id()
        token = request_id_var.set(request_id)
        request.state.request_id = request_id

        started = time.perf_counter()
        if self._metrics is not None:
            self._metrics.http_requests_in_flight.inc()

        try:
            response = await call_next(request)
        except Exception:
            # The exception handler will produce the response, but the metric
            # and the access line must still be recorded or a crashing endpoint
            # looks like an endpoint receiving no traffic.
            duration = time.perf_counter() - started
            self._record(request, status_code=500, duration=duration)
            raise
        finally:
            if self._metrics is not None:
                self._metrics.http_requests_in_flight.dec()
            request_id_var.reset(token)

        duration = time.perf_counter() - started
        response.headers[REQUEST_ID_HEADER] = request_id
        # Lets a client correlate its own observed latency with the server's.
        response.headers["X-Process-Time-Ms"] = f"{duration * 1000:.2f}"
        self._record(request, status_code=response.status_code, duration=duration)
        return response

    def _record(self, request: Request, *, status_code: int, duration: float) -> None:
        endpoint = _route_template(request)

        if self._metrics is not None:
            self._metrics.observe_request(
                method=request.method,
                endpoint=endpoint,
                status_code=status_code,
                duration_seconds=duration,
            )

        if not self._access_log or endpoint in _QUIET_PATHS:
            return

        # 5xx at error level so it is alertable; 4xx at info because a client
        # sending bad input is not a service problem.
        log = logger.info if status_code < 500 else logger.error
        log(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_ms": round(duration * 1000, 3),
                "client": request.client.host if request.client else None,
                "user_agent": request.headers.get("user-agent", "")[:200],
            },
        )


def _route_template(request: Request) -> str:
    """The matched route's path format, falling back to the raw path.

    The fallback only applies to unmatched routes (404s). Those are bounded in
    practice by the 404 handler, but to be safe the raw path is truncated so a
    scanner probing thousands of URLs cannot inflate cardinality without bound.
    """
    route = request.scope.get("route")
    path_format = getattr(route, "path_format", None) or getattr(route, "path", None)
    if path_format:
        return str(path_format)
    return request.url.path[:64] if request.url.path else "unknown"
