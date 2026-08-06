"""Liveness and readiness probes.

The distinction is the whole point, and getting it wrong is one of the most
common — and most damaging — Kubernetes misconfigurations.

``/health`` (liveness) answers **"is this process broken?"**. Failing it gets the
container **killed and restarted**. So it must not depend on the model, the
queue, or anything downstream. If the model fails to load, restarting does not
help — you get a crash loop that turns a recoverable degradation into an outage.
Liveness returns 200 as long as the event loop can serve a request.

``/ready`` (readiness) answers **"should traffic be routed here?"**. Failing it
removes the pod from the load balancer but leaves it running. This is where the
model, the batcher and the drain state belong. During shutdown it flips to
not-ready *first*, so the LB stops sending work before the process stops
accepting it — that ordering is what makes zero-downtime deploys possible.
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status

from .. import __version__
from ..dependencies import StateDep
from ..schemas import HealthResponse, ReadinessResponse

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    description=(
        "Returns 200 whenever the process is running and its event loop is "
        "responsive. Deliberately independent of model state: a failed model "
        "load should remove the pod from service, not restart it forever."
    ),
)
async def health(state: StateDep) -> HealthResponse:
    return HealthResponse(
        service=state.config.service_name,
        version=__version__,
        uptime_seconds=round(state.uptime_seconds, 3),
    )


# Kubernetes convention aliases, so a chart written either way works unchanged.
@router.get("/healthz", include_in_schema=False)
async def healthz(state: StateDep) -> HealthResponse:
    return await health(state)


@router.get("/live", include_in_schema=False)
async def live(state: StateDep) -> HealthResponse:
    return await health(state)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    responses={503: {"description": "Not ready to serve traffic."}},
    description=(
        "Returns 200 only when the model is loaded, the batching engine is "
        "running, and the service is not draining. Returns 503 otherwise, which "
        "removes the instance from the load balancer without restarting it."
    ),
)
async def ready(state: StateDep, response: Response) -> ReadinessResponse:
    checks = {
        "model_loaded": state.model is not None and state.ready,
        "batcher_running": state.batcher is not None and state.batcher.running,
        "not_shutting_down": not state.shutting_down,
    }
    all_ready = all(checks.values())

    if all_ready:
        return ReadinessResponse(status="ready", checks=checks)

    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    if state.shutting_down:
        detail = "service is draining in-flight requests before shutdown"
    elif not checks["model_loaded"]:
        detail = "model has not finished loading"
    else:
        detail = "batching engine is not running"
    return ReadinessResponse(status="not_ready", checks=checks, detail=detail)


@router.get("/readyz", include_in_schema=False)
async def readyz(state: StateDep, response: Response) -> ReadinessResponse:
    return await ready(state, response)


@router.get(
    "/startup",
    include_in_schema=False,
    summary="Startup probe",
)
async def startup(state: StateDep, response: Response) -> ReadinessResponse:
    """Startup probe, for orchestrators that support one.

    Semantically identical to readiness here, but pointing a separate startup
    probe at this path lets you give a slow model load a long timeout without
    also loosening the liveness timeout for the whole life of the pod.
    """
    return await ready(state, response)
