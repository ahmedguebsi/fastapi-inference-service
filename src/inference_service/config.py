"""Service configuration.

TOML via stdlib :mod:`tomllib`, layered with ``INFERENCE_*`` environment
variables. Environment wins, because that is how containers are configured and
the same image must run in every environment without a rebuild.

Unknown keys are fatal. A silently-ignored ``max_batch_sze`` typo means an
operator believes they tuned throughput when they did not.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ENV_PREFIX = "INFERENCE_"

DEFAULT_CONFIG_PATHS: tuple[str, ...] = (
    "/etc/inference-service/service.toml",
    "./config/service.toml",
)


class ConfigError(ValueError):
    """Raised when configuration is malformed. Always names the offending key."""


@dataclass(frozen=True, slots=True)
class ServerConfig:
    host: str = "0.0.0.0"  # noqa: S104 - containers must bind all interfaces
    port: int = 8000
    workers: int = 1
    # Must exceed the load balancer's idle timeout, otherwise the server closes
    # a connection the LB still believes is open and clients see sporadic 502s.
    # ALB defaults to 60s, so 65 is the conventional floor.
    keep_alive_timeout: int = 65
    # Upper bound on how long a graceful shutdown may take before uvicorn
    # force-closes. Must be less than the orchestrator's kill grace period.
    graceful_shutdown_timeout: float = 30.0

    # How long to keep serving after readiness flips false, before draining the
    # queue. This must exceed the load balancer's readiness probe period, or the
    # LB is still routing traffic when we stop accepting it -- which is exactly
    # the 502-during-deploy problem this whole sequence exists to prevent.
    #
    # Kubernetes' default probe period is 10s; an ALB's is 15-30s. Raise it in
    # production to match your LB. Set to 0 in tests, where there is no LB and
    # the delay is pure wall-clock waste.
    prestop_delay_s: float = 5.0

    root_path: str = ""

    def validate(self) -> None:
        if not 1 <= self.port <= 65535:
            raise ConfigError(f"server.port must be 1-65535, got {self.port}")
        if self.workers < 1:
            raise ConfigError(f"server.workers must be >= 1, got {self.workers}")
        if self.keep_alive_timeout < 1:
            raise ConfigError("server.keep_alive_timeout must be >= 1")
        if self.graceful_shutdown_timeout <= 0:
            raise ConfigError("server.graceful_shutdown_timeout must be > 0")
        if self.prestop_delay_s < 0:
            raise ConfigError("server.prestop_delay_s must be >= 0")
        if self.prestop_delay_s >= self.graceful_shutdown_timeout:
            raise ConfigError(
                f"server.prestop_delay_s ({self.prestop_delay_s}s) must be below "
                f"graceful_shutdown_timeout ({self.graceful_shutdown_timeout}s), "
                f"or there is no time left to drain the queue"
            )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    backend: str = "linear"
    artifact_path: str = "models/sentiment-v1.json"
    # Run a throwaway inference at startup so the first real request does not
    # absorb one-off initialisation cost as a latency spike.
    warmup: bool = True

    def validate(self) -> None:
        if self.backend not in ("linear",):
            raise ConfigError(f"model.backend must be one of ['linear'], got {self.backend!r}")
        if not self.artifact_path:
            raise ConfigError("model.artifact_path must not be empty")


@dataclass(frozen=True, slots=True)
class BatchingConfig:
    enabled: bool = True
    max_batch_size: int = 32
    # The latency/throughput dial. See docs/performance.md before changing it.
    max_wait_ms: float = 10.0
    max_queue_size: int = 512
    request_timeout_s: float = 10.0
    drain_timeout_s: float = 30.0

    def validate(self) -> None:
        if self.max_batch_size < 1:
            raise ConfigError("batching.max_batch_size must be >= 1")
        if self.max_wait_ms < 0:
            raise ConfigError("batching.max_wait_ms must be >= 0")
        if self.max_queue_size < 1:
            raise ConfigError("batching.max_queue_size must be >= 1")
        if self.request_timeout_s <= 0:
            raise ConfigError("batching.request_timeout_s must be > 0")
        if self.drain_timeout_s <= 0:
            raise ConfigError("batching.drain_timeout_s must be > 0")
        if self.max_wait_ms / 1000.0 >= self.request_timeout_s:
            raise ConfigError(
                f"batching.max_wait_ms ({self.max_wait_ms}ms) must be well below "
                f"request_timeout_s ({self.request_timeout_s}s), or every request "
                f"times out waiting to be batched"
            )


@dataclass(frozen=True, slots=True)
class APIConfig:
    max_texts_per_request: int = 64
    max_text_length: int = 4000
    # Empty means auth is disabled. Set via INFERENCE_API_KEYS in production.
    api_keys: tuple[str, ...] = ()
    cors_origins: tuple[str, ...] = ()
    docs_enabled: bool = True

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    def validate(self) -> None:
        if self.max_texts_per_request < 1:
            raise ConfigError("api.max_texts_per_request must be >= 1")
        if self.max_text_length < 1:
            raise ConfigError("api.max_text_length must be >= 1")
        if "*" in self.cors_origins and len(self.cors_origins) > 1:
            raise ConfigError("api.cors_origins cannot combine '*' with specific origins")


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    level: str = "INFO"
    format: str = "json"
    access_log: bool = True
    # Request bodies can contain user text. Off by default.
    log_request_body: bool = False

    def validate(self) -> None:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid:
            raise ConfigError(f"logging.level must be one of {sorted(valid)}, got {self.level!r}")
        if self.format not in ("json", "text"):
            raise ConfigError(f"logging.format must be 'json' or 'text', got {self.format!r}")


@dataclass(frozen=True, slots=True)
class ObservabilityConfig:
    metrics_enabled: bool = True
    metrics_path: str = "/metrics"
    # Buckets chosen for a service whose p99 target is ~100ms. Prometheus
    # histogram buckets are cumulative; too few and quantiles are meaningless,
    # too many and cardinality explodes.
    latency_buckets: tuple[float, ...] = (
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.075,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    )

    def validate(self) -> None:
        if not self.metrics_path.startswith("/"):
            raise ConfigError("observability.metrics_path must start with '/'")
        if not self.latency_buckets:
            raise ConfigError("observability.latency_buckets must not be empty")
        if list(self.latency_buckets) != sorted(self.latency_buckets):
            raise ConfigError("observability.latency_buckets must be in ascending order")


@dataclass(frozen=True, slots=True)
class Config:
    service_name: str = "inference-service"
    environment: str = "development"
    server: ServerConfig = field(default_factory=ServerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    batching: BatchingConfig = field(default_factory=BatchingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    source_path: str | None = None

    def validate(self) -> Config:
        self.server.validate()
        self.model.validate()
        self.batching.validate()
        self.api.validate()
        self.logging.validate()
        self.observability.validate()
        return self


def find_config_file(explicit: str | None = None) -> str | None:
    """First existing config path, or None. An explicit missing path is fatal."""
    if explicit:
        if not Path(explicit).is_file():
            raise ConfigError(f"config file not found: {explicit}")
        return explicit
    for candidate in DEFAULT_CONFIG_PATHS:
        if Path(candidate).is_file():
            return candidate
    return None


def load_config(path: str | None = None, *, env: Mapping[str, str] | None = None) -> Config:
    """Build a validated config from file plus environment overrides."""
    env = os.environ if env is None else env
    resolved = find_config_file(path or env.get(f"{ENV_PREFIX}CONFIG") or None)

    raw: dict[str, Any] = {}
    if resolved:
        try:
            with Path(resolved).open("rb") as handle:
                raw = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {resolved}: {exc}") from exc
        except OSError as exc:
            raise ConfigError(f"cannot read config {resolved}: {exc}") from exc

    config = _from_mapping(raw, source_path=resolved)
    return _apply_env(config, env).validate()


def _from_mapping(raw: Mapping[str, Any], *, source_path: str | None) -> Config:
    _reject_unknown(
        raw,
        {
            "service_name",
            "environment",
            "server",
            "model",
            "batching",
            "api",
            "logging",
            "observability",
        },
        "",
    )

    server_raw = _section(raw, "server")
    _reject_unknown(
        server_raw,
        {
            "host",
            "port",
            "workers",
            "keep_alive_timeout",
            "graceful_shutdown_timeout",
            "prestop_delay_s",
            "root_path",
        },
        "server",
    )
    server = ServerConfig(
        host=_str(server_raw, "host", "0.0.0.0"),  # noqa: S104
        port=_int(server_raw, "port", 8000),
        workers=_int(server_raw, "workers", 1),
        keep_alive_timeout=_int(server_raw, "keep_alive_timeout", 65),
        graceful_shutdown_timeout=_float(server_raw, "graceful_shutdown_timeout", 30.0),
        prestop_delay_s=_float(server_raw, "prestop_delay_s", 5.0),
        root_path=_str(server_raw, "root_path", ""),
    )

    model_raw = _section(raw, "model")
    _reject_unknown(model_raw, {"backend", "artifact_path", "warmup"}, "model")
    model = ModelConfig(
        backend=_str(model_raw, "backend", "linear"),
        artifact_path=_str(model_raw, "artifact_path", "models/sentiment-v1.json"),
        warmup=_bool(model_raw, "warmup", True),
    )

    batching_raw = _section(raw, "batching")
    _reject_unknown(
        batching_raw,
        {
            "enabled",
            "max_batch_size",
            "max_wait_ms",
            "max_queue_size",
            "request_timeout_s",
            "drain_timeout_s",
        },
        "batching",
    )
    batching = BatchingConfig(
        enabled=_bool(batching_raw, "enabled", True),
        max_batch_size=_int(batching_raw, "max_batch_size", 32),
        max_wait_ms=_float(batching_raw, "max_wait_ms", 10.0),
        max_queue_size=_int(batching_raw, "max_queue_size", 512),
        request_timeout_s=_float(batching_raw, "request_timeout_s", 10.0),
        drain_timeout_s=_float(batching_raw, "drain_timeout_s", 30.0),
    )

    api_raw = _section(raw, "api")
    _reject_unknown(
        api_raw,
        {"max_texts_per_request", "max_text_length", "api_keys", "cors_origins", "docs_enabled"},
        "api",
    )
    api = APIConfig(
        max_texts_per_request=_int(api_raw, "max_texts_per_request", 64),
        max_text_length=_int(api_raw, "max_text_length", 4000),
        api_keys=_str_tuple(api_raw, "api_keys"),
        cors_origins=_str_tuple(api_raw, "cors_origins"),
        docs_enabled=_bool(api_raw, "docs_enabled", True),
    )

    logging_raw = _section(raw, "logging")
    _reject_unknown(logging_raw, {"level", "format", "access_log", "log_request_body"}, "logging")
    logging_cfg = LoggingConfig(
        level=_str(logging_raw, "level", "INFO").upper(),
        format=_str(logging_raw, "format", "json"),
        access_log=_bool(logging_raw, "access_log", True),
        log_request_body=_bool(logging_raw, "log_request_body", False),
    )

    obs_raw = _section(raw, "observability")
    _reject_unknown(
        obs_raw, {"metrics_enabled", "metrics_path", "latency_buckets"}, "observability"
    )
    default_buckets = ObservabilityConfig().latency_buckets
    observability = ObservabilityConfig(
        metrics_enabled=_bool(obs_raw, "metrics_enabled", True),
        metrics_path=_str(obs_raw, "metrics_path", "/metrics"),
        latency_buckets=_float_tuple(obs_raw, "latency_buckets") or default_buckets,
    )

    return Config(
        service_name=_str(raw, "service_name", "inference-service"),
        environment=_str(raw, "environment", "development"),
        server=server,
        model=model,
        batching=batching,
        api=api,
        logging=logging_cfg,
        observability=observability,
        source_path=source_path,
    )


def _apply_env(config: Config, env: Mapping[str, str]) -> Config:
    """Apply ``INFERENCE_*`` overrides."""

    def get(name: str) -> str | None:
        return env.get(f"{ENV_PREFIX}{name}")

    server, model, batching = config.server, config.model, config.batching
    api, logging_cfg, obs = config.api, config.logging, config.observability
    service_name, environment = config.service_name, config.environment

    if (v := get("SERVICE_NAME")) is not None:
        service_name = v
    if (v := get("ENVIRONMENT")) is not None:
        environment = v

    if (v := get("HOST")) is not None:
        server = replace(server, host=v)
    if (v := get("PORT")) is not None:
        server = replace(server, port=_parse_int(v, "PORT"))
    if (v := get("WORKERS")) is not None:
        server = replace(server, workers=_parse_int(v, "WORKERS"))
    if (v := get("GRACEFUL_SHUTDOWN_TIMEOUT")) is not None:
        server = replace(
            server, graceful_shutdown_timeout=_parse_float(v, "GRACEFUL_SHUTDOWN_TIMEOUT")
        )
    if (v := get("PRESTOP_DELAY_S")) is not None:
        server = replace(server, prestop_delay_s=_parse_float(v, "PRESTOP_DELAY_S"))
    if (v := get("ROOT_PATH")) is not None:
        server = replace(server, root_path=v)

    if (v := get("MODEL_BACKEND")) is not None:
        model = replace(model, backend=v)
    if (v := get("MODEL_PATH")) is not None:
        model = replace(model, artifact_path=v)
    if (v := get("MODEL_WARMUP")) is not None:
        model = replace(model, warmup=_parse_bool(v, "MODEL_WARMUP"))

    if (v := get("BATCHING_ENABLED")) is not None:
        batching = replace(batching, enabled=_parse_bool(v, "BATCHING_ENABLED"))
    if (v := get("MAX_BATCH_SIZE")) is not None:
        batching = replace(batching, max_batch_size=_parse_int(v, "MAX_BATCH_SIZE"))
    if (v := get("MAX_WAIT_MS")) is not None:
        batching = replace(batching, max_wait_ms=_parse_float(v, "MAX_WAIT_MS"))
    if (v := get("MAX_QUEUE_SIZE")) is not None:
        batching = replace(batching, max_queue_size=_parse_int(v, "MAX_QUEUE_SIZE"))
    if (v := get("REQUEST_TIMEOUT_S")) is not None:
        batching = replace(batching, request_timeout_s=_parse_float(v, "REQUEST_TIMEOUT_S"))

    if (v := get("API_KEYS")) is not None:
        api = replace(api, api_keys=_split_list(v))
    if (v := get("CORS_ORIGINS")) is not None:
        api = replace(api, cors_origins=_split_list(v))
    if (v := get("DOCS_ENABLED")) is not None:
        api = replace(api, docs_enabled=_parse_bool(v, "DOCS_ENABLED"))
    if (v := get("MAX_TEXTS_PER_REQUEST")) is not None:
        api = replace(api, max_texts_per_request=_parse_int(v, "MAX_TEXTS_PER_REQUEST"))

    if (v := get("LOG_LEVEL")) is not None:
        logging_cfg = replace(logging_cfg, level=v.upper())
    if (v := get("LOG_FORMAT")) is not None:
        logging_cfg = replace(logging_cfg, format=v.lower())
    if (v := get("ACCESS_LOG")) is not None:
        logging_cfg = replace(logging_cfg, access_log=_parse_bool(v, "ACCESS_LOG"))

    if (v := get("METRICS_ENABLED")) is not None:
        obs = replace(obs, metrics_enabled=_parse_bool(v, "METRICS_ENABLED"))

    return replace(
        config,
        service_name=service_name,
        environment=environment,
        server=server,
        model=model,
        batching=batching,
        api=api,
        logging=logging_cfg,
        observability=obs,
    )


# -- typed accessors -------------------------------------------------------
def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {})
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{name}] must be a table, got {type(value).__name__}")
    return value


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], section: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        where = f"[{section}]" if section else "top level"
        raise ConfigError(
            f"unknown configuration key(s) at {where}: {sorted(unknown)}; "
            f"allowed: {sorted(allowed)}"
        )


def _str(raw: Mapping[str, Any], key: str, default: str) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str):
        raise ConfigError(f"{key} must be a string, got {type(value).__name__}")
    return value


def _bool(raw: Mapping[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a boolean, got {type(value).__name__}")
    return value


def _int(raw: Mapping[str, Any], key: str, default: int) -> int:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{key} must be an integer, got {type(value).__name__}")
    return value


def _float(raw: Mapping[str, Any], key: str, default: float) -> float:
    value = raw.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{key} must be a number, got {type(value).__name__}")
    return float(value)


def _str_tuple(raw: Mapping[str, Any], key: str) -> tuple[str, ...]:
    value = raw.get(key, ())
    if isinstance(value, str):
        raise ConfigError(f"{key} must be an array of strings, not a bare string")
    if not isinstance(value, Sequence):
        raise ConfigError(f"{key} must be an array of strings, got {type(value).__name__}")
    for item in value:
        if not isinstance(item, str):
            raise ConfigError(f"{key} entries must be strings, got {type(item).__name__}")
    return tuple(value)


def _float_tuple(raw: Mapping[str, Any], key: str) -> tuple[float, ...]:
    value = raw.get(key, ())
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise ConfigError(f"{key} must be an array of numbers")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{key} entries must be numbers, got {type(item).__name__}")
    return tuple(float(item) for item in value)


def _parse_bool(value: str, key: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ConfigError(f"{ENV_PREFIX}{key} must be boolean-like, got {value!r}")


def _parse_int(value: str, key: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{key} must be an integer, got {value!r}") from exc


def _parse_float(value: str, key: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{ENV_PREFIX}{key} must be a number, got {value!r}") from exc


def _split_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())
