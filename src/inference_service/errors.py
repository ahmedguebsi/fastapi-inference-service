"""Error taxonomy and exception handlers.

Two rules drive everything here:

1. **Every error response has the same shape.** Clients parse one envelope, not
   FastAPI's default for validation errors plus something else for ours.
2. **Internal detail never reaches the client.** A stack trace or a filesystem
   path in a 500 response is an information leak. It goes to the logs, keyed by
   request id, and the client gets that id to quote in a support ticket.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .logging_setup import get_logger, request_id_var
from .schemas import ErrorDetail, ErrorResponse

logger = get_logger("errors")


class ServiceError(Exception):
    """Base class for errors the service raises deliberately."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"
    message: str = "An internal error occurred."

    def __init__(
        self,
        message: str | None = None,
        *,
        details: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.message = message or self.message
        self.details = details
        self.headers = headers
        super().__init__(self.message)


class ModelNotReadyError(ServiceError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "model_not_ready"
    message = "The model is not loaded yet."


class QueueFullServiceError(ServiceError):
    """Backpressure. 503 plus Retry-After is the honest answer under overload.

    Returning 503 quickly is strictly better than queueing without bound: the
    client can retry against another replica, and the autoscaler sees the signal.
    """

    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "queue_full"
    message = "The service is at capacity. Retry shortly."


class InferenceTimeoutServiceError(ServiceError):
    status_code = status.HTTP_504_GATEWAY_TIMEOUT
    code = "inference_timeout"
    message = "Inference did not complete within the allotted time."


class ShuttingDownError(ServiceError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    code = "shutting_down"
    message = "The service is shutting down and is not accepting new requests."


class InferenceFailedError(ServiceError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    code = "inference_failed"
    message = "Inference failed."


class UnauthorizedError(ServiceError):
    status_code = status.HTTP_401_UNAUTHORIZED
    code = "unauthorized"
    message = "A valid API key is required."


class PayloadTooLargeError(ServiceError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    code = "payload_too_large"
    message = "The request payload exceeds the configured limit."


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id_var.get("") or None,
            details=details,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=body.model_dump(exclude_none=True),
        headers=headers,
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Install handlers so every failure path returns the same envelope."""

    # Backpressure, timeouts and drain rejections are 5xx, but they are the
    # design working as intended under load -- not defects. Logging them at
    # ERROR would make any "error rate" dashboard spike during exactly the
    # situation where you need a clear signal, and is how alert fatigue starts.
    expected_codes = frozenset(
        {
            QueueFullServiceError.code,
            InferenceTimeoutServiceError.code,
            ShuttingDownError.code,
            ModelNotReadyError.code,
        }
    )

    @app.exception_handler(ServiceError)
    async def handle_service_error(_: Request, exc: ServiceError) -> JSONResponse:
        expected = exc.status_code < 500 or exc.code in expected_codes
        log = logger.warning if expected else logger.error
        log(
            exc.message,
            extra={"error_code": exc.code, "status_code": exc.status_code},
        )
        return _error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
            details=exc.details,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's raw errors can embed the offending input, which for this
        # service is user text. Only the location and reason are echoed back.
        fields = [
            {
                "field": ".".join(str(part) for part in error.get("loc", ())[1:]) or "body",
                "reason": error.get("msg", "invalid"),
                "type": error.get("type", "unknown"),
            }
            for error in exc.errors()
        ]
        logger.info("request validation failed", extra={"fields": fields})
        return _error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            code="validation_error",
            message="The request body failed validation.",
            details={"fields": fields},
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {
            401: "unauthorized",
            403: "forbidden",
            404: "not_found",
            405: "method_not_allowed",
            413: "payload_too_large",
            429: "rate_limited",
        }
        return _error_response(
            status_code=exc.status_code,
            code=codes.get(exc.status_code, f"http_{exc.status_code}"),
            message=str(exc.detail),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_: Request, exc: Exception) -> JSONResponse:
        # The catch-all. The exception and stack go to the logs with the request
        # id; the client gets the id and nothing else.
        logger.exception("unhandled exception", extra={"error_type": type(exc).__name__})
        return _error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            code="internal_error",
            message=("An internal error occurred. Quote the request_id when reporting this."),
        )
