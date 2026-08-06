"""Service introspection and metrics exposition."""

from __future__ import annotations

from fastapi import APIRouter, Response

from .. import __version__
from ..dependencies import StateDep
from ..schemas import BatcherStatsResponse, MetaResponse

router = APIRouter(tags=["meta"])


@router.get(
    "/v1/meta",
    response_model=MetaResponse,
    summary="Service and model identity",
    description=(
        "Returns the loaded model's identity (including the artifact SHA-256), "
        "the effective batching configuration, and live scheduler statistics. "
        "The first question in any inference incident is 'which model is "
        "actually loaded?' — this answers it without shell access."
    ),
)
async def meta(state: StateDep) -> MetaResponse:
    batcher = state.batcher
    stats = batcher.stats() if batcher is not None else None
    return MetaResponse(
        service=state.config.service_name,
        version=__version__,
        environment=state.config.environment,
        model=(
            {
                **state.model.info.to_dict(),
                "load_duration_seconds": round(state.model_load_seconds, 4),
            }
            if state.model is not None
            else {"status": "not_loaded"}
        ),
        batching={
            "enabled": state.config.batching.enabled,
            "max_batch_size": state.config.batching.max_batch_size,
            "max_wait_ms": state.config.batching.max_wait_ms,
            "max_queue_size": state.config.batching.max_queue_size,
            "request_timeout_s": state.config.batching.request_timeout_s,
        },
        batcher_stats=(
            BatcherStatsResponse(
                queue_depth=stats.queue_depth,
                queue_capacity=stats.queue_capacity,
                batches_processed=stats.batches_processed,
                items_processed=stats.items_processed,
                items_rejected=stats.items_rejected,
                items_timed_out=stats.items_timed_out,
                mean_batch_size=stats.mean_batch_size,
                running=stats.running,
            )
            if stats is not None
            else BatcherStatsResponse(
                queue_depth=0,
                queue_capacity=0,
                batches_processed=0,
                items_processed=0,
                items_rejected=0,
                items_timed_out=0,
                mean_batch_size=0.0,
                running=False,
            )
        ),
    )


def build_metrics_router(path: str) -> APIRouter:
    """Metrics endpoint, mounted at the configured path.

    Excluded from the OpenAPI schema: it is an operational endpoint scraped by
    Prometheus, not part of the API contract clients generate against.
    """
    metrics_router = APIRouter(tags=["meta"])

    @metrics_router.get(path, include_in_schema=False)
    async def metrics_endpoint(state: StateDep) -> Response:
        payload, content_type = state.metrics.render()
        return Response(content=payload, media_type=content_type)

    return metrics_router
