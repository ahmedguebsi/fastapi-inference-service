"""Inference endpoints.

Handlers stay thin on purpose: validate, submit to the batching engine, map
backend exceptions onto the HTTP error taxonomy, record metrics. All scheduling
logic lives in :mod:`inference_service.serving.batcher`, so it can be tested
without an HTTP client and reasoned about without FastAPI in the picture.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, status

from ..dependencies import AuthDep, BatcherDep, ConfigDep, MetricsDep
from ..errors import (
    InferenceFailedError,
    InferenceTimeoutServiceError,
    PayloadTooLargeError,
    QueueFullServiceError,
    ShuttingDownError,
)
from ..logging_setup import get_logger
from ..schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    PredictionResult,
    PredictRequest,
    PredictResponse,
)
from ..serving.batcher import (
    BatcherClosedError,
    InferenceTimeoutError,
    QueueFullError,
)

logger = get_logger("predict")

router = APIRouter(prefix="/v1", tags=["inference"])

# Documented on both endpoints so generated clients know the failure modes.
ERROR_RESPONSES: dict[int | str, dict[str, object]] = {
    status.HTTP_401_UNAUTHORIZED: {
        "model": ErrorResponse,
        "description": "Missing or invalid API key.",
    },
    status.HTTP_413_CONTENT_TOO_LARGE: {
        "model": ErrorResponse,
        "description": "Payload exceeds the configured limit.",
    },
    status.HTTP_422_UNPROCESSABLE_CONTENT: {
        "model": ErrorResponse,
        "description": "Request body failed validation.",
    },
    status.HTTP_500_INTERNAL_SERVER_ERROR: {
        "model": ErrorResponse,
        "description": "Inference failed.",
    },
    status.HTTP_503_SERVICE_UNAVAILABLE: {
        "model": ErrorResponse,
        "description": "Model not ready, at capacity, or draining.",
    },
    status.HTTP_504_GATEWAY_TIMEOUT: {
        "model": ErrorResponse,
        "description": "Inference exceeded the time budget.",
    },
}

# Advises the client how long to back off. Matching the batching window's order
# of magnitude means a retry arrives after the queue has plausibly drained.
RETRY_AFTER_SECONDS = "1"


def _translate(exc: Exception) -> Exception:
    """Map batching-layer failures onto the HTTP error taxonomy.

    Kept in one place so both endpoints answer identically, and so the batcher
    never needs to know that HTTP exists.
    """
    if isinstance(exc, QueueFullError):
        return QueueFullServiceError(str(exc), headers={"Retry-After": RETRY_AFTER_SECONDS})
    if isinstance(exc, InferenceTimeoutError):
        return InferenceTimeoutServiceError(str(exc))
    if isinstance(exc, BatcherClosedError):
        return ShuttingDownError(str(exc), headers={"Retry-After": RETRY_AFTER_SECONDS})
    # Anything else is a genuine backend failure. The message is logged, not
    # returned, so model internals do not leak to callers.
    logger.exception("inference backend error", extra={"error_type": type(exc).__name__})
    return InferenceFailedError()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Classify a single text",
    responses=ERROR_RESPONSES,
    description=(
        "Submits one text for classification. The request is placed on the "
        "shared inference queue and grouped with concurrent requests into a "
        "single model call."
    ),
)
async def predict(
    payload: PredictRequest,
    request: Request,
    batcher: BatcherDep,
    config: ConfigDep,
    metrics: MetricsDep,
    _auth: AuthDep,
) -> PredictResponse:
    if len(payload.text) > config.api.max_text_length:
        raise PayloadTooLargeError(
            f"text exceeds the {config.api.max_text_length} character limit",
            details={"limit": config.api.max_text_length, "received": len(payload.text)},
        )

    started = time.perf_counter()
    try:
        prediction = await batcher.predict(payload.text)
    except Exception as exc:
        raise _translate(exc) from exc

    latency_ms = (time.perf_counter() - started) * 1000.0
    metrics.observe_predictions([prediction.label])

    model_info = batcher.model.info
    return PredictResponse(
        result=PredictionResult(**prediction.to_dict()),
        model_name=model_info.name,
        model_version=model_info.version,
        request_id=getattr(request.state, "request_id", ""),
        latency_ms=round(latency_ms, 3),
    )


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Classify several texts",
    responses=ERROR_RESPONSES,
    description=(
        "Submits several texts in one call. Note this is a client-side "
        "convenience: each text is scheduled independently, so a large request "
        "cannot monopolise an inference batch and starve concurrent callers."
    ),
)
async def predict_batch(
    payload: BatchPredictRequest,
    request: Request,
    batcher: BatcherDep,
    config: ConfigDep,
    metrics: MetricsDep,
    _auth: AuthDep,
) -> BatchPredictResponse:
    if len(payload.texts) > config.api.max_texts_per_request:
        raise PayloadTooLargeError(
            f"request contains {len(payload.texts)} texts; the limit is "
            f"{config.api.max_texts_per_request}",
            details={
                "limit": config.api.max_texts_per_request,
                "received": len(payload.texts),
            },
        )
    for index, text in enumerate(payload.texts):
        if len(text) > config.api.max_text_length:
            raise PayloadTooLargeError(
                f"texts[{index}] exceeds the {config.api.max_text_length} character limit",
                details={"index": index, "limit": config.api.max_text_length},
            )

    started = time.perf_counter()
    try:
        predictions = await batcher.predict_many(payload.texts)
    except Exception as exc:
        raise _translate(exc) from exc

    latency_ms = (time.perf_counter() - started) * 1000.0
    metrics.observe_predictions([p.label for p in predictions])

    model_info = batcher.model.info
    return BatchPredictResponse(
        results=[PredictionResult(**p.to_dict()) for p in predictions],
        model_name=model_info.name,
        model_version=model_info.version,
        request_id=getattr(request.state, "request_id", ""),
        latency_ms=round(latency_ms, 3),
        count=len(predictions),
    )
