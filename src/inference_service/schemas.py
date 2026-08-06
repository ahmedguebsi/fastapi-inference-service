"""Request and response models.

Pydantic v2 does double duty here: runtime validation at the edge, and the
OpenAPI schema clients generate from. Limits (text length, batch size) are
enforced *here*, before anything reaches the queue, so a malicious or buggy
client cannot consume inference capacity with input the service was never going
to accept.

Every model sets ``extra="forbid"``. A client sending ``{"txt": ...}`` instead
of ``{"text": ...}`` gets a 422 naming the problem, rather than a confusing
"field required" for a field they believe they sent.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Defaults mirror config.APIConfig. The config values are the operational
# authority and are enforced in the router; these bound the schema so OpenAPI
# documents something sensible and obviously-bad input dies at parse time.
MAX_TEXT_LENGTH = 4000
MAX_TEXTS_PER_REQUEST = 64

TextField = Annotated[
    str,
    Field(
        min_length=1,
        max_length=MAX_TEXT_LENGTH,
        description="Input text to classify.",
        examples=["This service is fantastic and I would recommend it."],
    ),
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)


class PredictRequest(StrictModel):
    """Single-input prediction."""

    text: TextField

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        """Reject whitespace-only input.

        ``min_length=1`` passes for "   ", which produces no tokens and yields a
        meaningless prediction at the model's bias. Better to reject it with a
        clear message than return a confident-looking result derived from
        nothing.
        """
        if not value.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return value

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"text": "This service is fantastic and I would recommend it."}]
        },
    )


class BatchPredictRequest(StrictModel):
    """Multiple inputs in one call.

    Note this is a client-side convenience, not the batching mechanism. The
    server submits each text to the scheduler independently, so a 64-item
    request cannot monopolise a batch and starve concurrent single-item callers.
    """

    texts: Annotated[
        list[TextField],
        Field(
            min_length=1,
            max_length=MAX_TEXTS_PER_REQUEST,
            description="Input texts to classify.",
        ),
    ]

    @field_validator("texts")
    @classmethod
    def none_blank(cls, value: list[str]) -> list[str]:
        for index, text in enumerate(value):
            if not text.strip():
                raise ValueError(
                    f"texts[{index}] must contain at least one non-whitespace character"
                )
        return value

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{"texts": ["Excellent, exactly what I needed.", "Broken on arrival."]}]
        },
    )


class PredictionResult(StrictModel):
    """One classification result."""

    label: str = Field(description="Highest-scoring label.")
    score: float = Field(ge=0.0, le=1.0, description="Confidence in `label`.")
    scores: dict[str, float] = Field(description="Score for every label.")


class PredictResponse(StrictModel):
    """Response for a single-input prediction."""

    result: PredictionResult
    model_name: str
    model_version: str
    request_id: str
    latency_ms: float = Field(description="Server-side handling time.")

    # Pydantic reserves the `model_` prefix for its own API. Without this,
    # `model_name` and `model_version` emit a warning at class-creation time.
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class BatchPredictResponse(StrictModel):
    """Response for a multi-input prediction."""

    results: list[PredictionResult]
    model_name: str
    model_version: str
    request_id: str
    latency_ms: float
    count: int

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class HealthResponse(StrictModel):
    """Liveness. Deliberately carries no dependency state."""

    status: Literal["ok"] = "ok"
    service: str
    version: str
    uptime_seconds: float


class ReadinessResponse(StrictModel):
    """Readiness, including why the service is or is not ready."""

    status: Literal["ready", "not_ready"]
    checks: dict[str, bool]
    detail: str | None = None


class BatcherStatsResponse(StrictModel):
    queue_depth: int
    queue_capacity: int
    batches_processed: int
    items_processed: int
    items_rejected: int
    items_timed_out: int
    mean_batch_size: float
    running: bool


class MetaResponse(StrictModel):
    """Service and model identity, for debugging and support."""

    service: str
    version: str
    environment: str
    model: dict[str, Any]
    batching: dict[str, Any]
    batcher_stats: BatcherStatsResponse

    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ErrorDetail(StrictModel):
    """Every error response has this shape.

    A single machine-readable ``code`` means clients can branch on the failure
    mode without string-matching a human message that will be reworded later.
    """

    code: str = Field(description="Stable, machine-readable error identifier.")
    message: str = Field(description="Human-readable explanation.")
    request_id: str | None = Field(default=None, description="Correlates with server logs.")
    details: dict[str, Any] | None = None


class ErrorResponse(StrictModel):
    error: ErrorDetail
