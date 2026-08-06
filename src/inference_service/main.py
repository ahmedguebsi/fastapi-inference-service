"""Application factory and lifecycle.

## Why a factory rather than a module-level ``app``

A module-level ``app = FastAPI()`` is created at import time, which means tests
share one instance, one metrics registry, and one config. ``create_app()`` lets
the suite build isolated applications with different settings in the same
process — which is what makes the auth, backpressure and shutdown tests possible
at all.

## Startup and shutdown ordering

The order below is the part that matters in production, and it is worth reading
carefully:

**Startup**

1. Load the model. Fatal on failure — better a crash with a clear message than a
   pod that sits in the rotation answering 503.
2. Warm up, so the first real request does not pay one-off initialisation cost.
3. Start the batching engine.
4. *Only then* flip ``ready``. The readiness probe gates the load balancer, so
   flipping it earlier would route traffic at a service that cannot serve it.

**Shutdown** is the reverse, and the first step is the one people miss:

1. Flip ``ready`` to false **immediately**. The load balancer notices on its
   next probe and stops sending new work. This happens *before* anything is torn
   down, and it is what makes a rolling deploy lose zero requests.
2. Wait out the LB's probe interval, so in-flight routing decisions land.
3. Drain the queue: finish accepted work, reject new submissions.
4. Close the model.

Skipping step 1 — tearing down first and de-registering later — is the single
most common cause of 502s during deploys.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware

from . import __version__
from .config import Config, load_config
from .dependencies import AppState
from .errors import register_exception_handlers
from .logging_setup import configure_logging, get_logger
from .metrics import ServiceMetrics
from .middleware import ObservabilityMiddleware
from .model.loader import load_backend
from .routers import health, meta, predict
from .serving.batcher import DynamicBatcher

logger = get_logger("main")

DESCRIPTION = """
Production-grade text classification inference service.

Requests are grouped by a dynamic batching scheduler into single model calls,
with bounded queueing, per-request timeouts and backpressure. Every response
carries an `X-Request-ID` that correlates with the structured logs.

* `POST /v1/predict` — classify one text
* `POST /v1/predict/batch` — classify several
* `GET /health` — liveness (never depends on the model)
* `GET /ready` — readiness (gates the load balancer)
* `GET /v1/meta` — loaded model identity and live scheduler stats
* `GET /metrics` — Prometheus exposition
"""


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Own the model and batcher for the lifetime of the application."""
    state: AppState = app.state.app_state
    config = state.config

    logger.info(
        "starting up",
        extra={
            "service": config.service_name,
            "version": __version__,
            "environment": config.environment,
            "config_source": config.source_path or "<defaults>",
        },
    )

    load_started = time.perf_counter()
    try:
        model = load_backend(config.model)
    except Exception:
        # Deliberately not caught: propagating aborts startup, which is the
        # correct outcome. Logged here so the reason is the first thing in the
        # container log rather than a bare traceback.
        logger.exception(
            "model load failed; refusing to start",
            extra={"artifact_path": config.model.artifact_path},
        )
        raise

    state.model_load_seconds = time.perf_counter() - load_started
    state.model = model
    state.metrics.model_load_duration.set(state.model_load_seconds)
    state.metrics.model_info.info(
        {
            "name": model.info.name,
            "version": model.info.version,
            "backend": model.info.backend,
            "task": model.info.task,
            "artifact_sha256": model.info.artifact_sha256 or "",
        }
    )
    state.metrics.build_info.info(
        {
            "version": __version__,
            "service": config.service_name,
            "environment": config.environment,
        }
    )

    batcher = DynamicBatcher(
        model,
        max_batch_size=config.batching.max_batch_size if config.batching.enabled else 1,
        max_wait_ms=config.batching.max_wait_ms if config.batching.enabled else 0.0,
        max_queue_size=config.batching.max_queue_size,
        request_timeout_s=config.batching.request_timeout_s,
        metrics=state.metrics,
    )
    await batcher.start()
    state.batcher = batcher

    # Ready last: everything above must succeed before traffic is welcome.
    state.ready = True
    state.metrics.set_ready(True)
    state.metrics.start_time.set(time.time())

    # Announce the shutdown budget at startup, while someone is looking at the
    # logs. Shutdown takes prestop + drain; if the orchestrator's kill grace is
    # shorter, the process is SIGKILLed mid-drain and in-flight requests are
    # lost — and the symptom (exit code 137 during deploys) is easy to
    # misdiagnose as a crash. Docker's default `docker stop` grace is 10s and
    # Kubernetes' terminationGracePeriodSeconds is 30s, so this trips easily.
    shutdown_budget = config.server.prestop_delay_s + config.batching.drain_timeout_s
    logger.info(
        "ready to serve",
        extra={
            "model": model.info.name,
            "model_version": model.info.version,
            "load_seconds": round(state.model_load_seconds, 4),
            "shutdown_budget_seconds": shutdown_budget,
        },
    )
    if shutdown_budget > 10:
        logger.warning(
            "graceful shutdown may take up to %.0fs (prestop %.0fs + drain %.0fs). "
            "Set the orchestrator's kill grace ABOVE this or the process is "
            "SIGKILLed mid-drain: docker `--stop-timeout`/compose "
            "`stop_grace_period`, Kubernetes `terminationGracePeriodSeconds`.",
            shutdown_budget,
            config.server.prestop_delay_s,
            config.batching.drain_timeout_s,
            extra={"shutdown_budget_seconds": shutdown_budget},
        )

    try:
        yield
    finally:
        logger.info("shutting down")

        # 1. Stop advertising readiness FIRST, so the load balancer drains us
        #    while we are still able to serve what it has already routed.
        state.shutting_down = True
        state.ready = False
        state.metrics.set_ready(False)

        prestop = config.server.prestop_delay_s
        if prestop > 0:
            logger.info("awaiting load balancer deregistration", extra={"seconds": prestop})
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.sleep(prestop)

        # 2. Finish accepted work.
        await batcher.stop(drain_timeout_s=config.batching.drain_timeout_s)

        # 3. Release model resources.
        with contextlib.suppress(Exception):
            model.close()

        logger.info("shutdown complete")


def create_app(config: Config | None = None, *, metrics: ServiceMetrics | None = None) -> FastAPI:
    """Build a fully wired application."""
    config = config or load_config()
    configure_logging(
        config.logging,
        service_name=config.service_name,
        environment=config.environment,
    )

    service_metrics = metrics or ServiceMetrics(config.observability)

    app = FastAPI(
        title=config.service_name,
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
        root_path=config.server.root_path,
        # Interactive docs are convenient in development and an unnecessary
        # surface in production, where clients use the generated OpenAPI spec.
        docs_url="/docs" if config.api.docs_enabled else None,
        redoc_url="/redoc" if config.api.docs_enabled else None,
        openapi_url="/openapi.json" if config.api.docs_enabled else None,
    )

    app.state.app_state = AppState(config=config, metrics=service_metrics)

    # Middleware runs in reverse registration order, so ObservabilityMiddleware
    # is added last and therefore runs first -- it must wrap everything to time
    # the full request and assign the id before any other layer logs.
    if config.api.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(config.api.cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
            expose_headers=["X-Request-ID", "X-Process-Time-Ms"],
        )

    # Prediction responses for large batches compress well. minimum_size avoids
    # burning CPU on small bodies where compression is a net loss.
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    app.add_middleware(
        ObservabilityMiddleware,
        config=config,
        metrics=service_metrics if config.observability.metrics_enabled else None,
    )

    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(predict.router)
    app.include_router(meta.router)
    if config.observability.metrics_enabled:
        app.include_router(meta.build_metrics_router(config.observability.metrics_path))

    return app


# Uvicorn's factory mode calls this: `uvicorn inference_service.main:app --factory`
app = create_app
