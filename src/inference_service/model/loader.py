"""Model backend construction.

A single factory keeps the "which backend?" decision in one reviewable place and
makes adding a backend a two-line change rather than a hunt through the
application wiring.
"""

from __future__ import annotations

from pathlib import Path

from ..config import ModelConfig
from ..logging_setup import get_logger
from .base import ModelBackend, ModelLoadError
from .linear import LinearTextClassifier

logger = get_logger("model.loader")

# Registered backends. An ONNX backend would be added here alongside an
# optional dependency group; see docs/architecture.md.
BACKENDS: dict[str, type[ModelBackend]] = {
    LinearTextClassifier.BACKEND_NAME: LinearTextClassifier,
}


def build_backend(config: ModelConfig) -> ModelBackend:
    """Instantiate the configured backend. Does not load the artifact."""
    backend_cls = BACKENDS.get(config.backend)
    if backend_cls is None:
        raise ModelLoadError(
            f"unknown model backend {config.backend!r}; available: {sorted(BACKENDS)}"
        )

    artifact = Path(config.artifact_path)
    if not artifact.is_absolute():
        artifact = Path.cwd() / artifact

    logger.info(
        "building model backend",
        extra={"backend": config.backend, "artifact_path": str(artifact)},
    )
    return backend_cls(artifact)  # type: ignore[call-arg]


def load_backend(config: ModelConfig) -> ModelBackend:
    """Build, load and optionally warm up a backend.

    Failures here are intentionally fatal to startup. A service that boots
    without a model and answers every request with 503 is harder to diagnose
    than one that refuses to start with a clear message in its first log line —
    and in an orchestrator, a crash at startup is visible immediately, whereas a
    silently model-less pod can sit in a rotation looking healthy.
    """
    backend = build_backend(config)
    backend.load()
    info = backend.info
    logger.info(
        "model loaded",
        extra={
            # Not "name": that collides with LogRecord.name. The adapter would
            # rename it, but an explicit field name is clearer in the log.
            "model_name": info.name,
            "model_version": info.version,
            "backend": info.backend,
            "parameters": info.parameters,
            "artifact_sha256": info.artifact_sha256,
        },
    )
    if config.warmup:
        backend.warmup()
        logger.info("model warmup complete")
    return backend
