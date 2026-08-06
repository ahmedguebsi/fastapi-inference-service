"""Shared FastAPI dependencies and application state.

State lives on ``app.state`` and is reached through dependencies rather than
module-level globals. That is what allows the test suite to build several
independent apps in one process — with different configs and isolated metric
registries — instead of fighting shared mutable state.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import APIKeyHeader

from .config import Config
from .errors import ModelNotReadyError, ShuttingDownError, UnauthorizedError
from .metrics import ServiceMetrics
from .model.base import ModelBackend
from .serving.batcher import DynamicBatcher

API_KEY_HEADER_NAME = "X-API-Key"

# auto_error=False so a missing key reaches our handler and returns the standard
# error envelope, rather than FastAPI's own 403 with a different body shape.
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


@dataclass(slots=True)
class AppState:
    """Everything the request path needs, assembled during startup."""

    config: Config
    metrics: ServiceMetrics
    model: ModelBackend | None = None
    batcher: DynamicBatcher | None = None
    ready: bool = False
    shutting_down: bool = False
    started_at: float = field(default_factory=time.time)
    model_load_seconds: float = 0.0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at


def get_state(request: Request) -> AppState:
    state: AppState = request.app.state.app_state
    return state


def get_config(state: Annotated[AppState, Depends(get_state)]) -> Config:
    return state.config


def get_metrics(state: Annotated[AppState, Depends(get_state)]) -> ServiceMetrics:
    return state.metrics


def get_batcher(state: Annotated[AppState, Depends(get_state)]) -> DynamicBatcher:
    """The batching engine, or 503 if the service is not serving.

    Both conditions matter. ``not ready`` covers startup before the model is
    loaded; ``shutting_down`` covers the drain window, where the process is
    still accepting connections but must not take new work.
    """
    if state.shutting_down:
        raise ShuttingDownError
    if not state.ready or state.batcher is None:
        raise ModelNotReadyError
    return state.batcher


def get_model(state: Annotated[AppState, Depends(get_state)]) -> ModelBackend:
    if not state.ready or state.model is None:
        raise ModelNotReadyError
    return state.model


async def require_api_key(
    request: Request,
    api_key: Annotated[str | None, Depends(api_key_header)],
) -> None:
    """Authenticate when API keys are configured; no-op otherwise.

    Comparison uses :func:`secrets.compare_digest`, which takes constant time
    with respect to the number of matching leading characters. A plain ``==``
    short-circuits on the first difference and leaks key material to an attacker
    who can measure response times precisely enough.
    """
    state: AppState = request.app.state.app_state
    configured = state.config.api.api_keys
    if not configured:
        return

    if not api_key:
        raise UnauthorizedError(
            f"Missing {API_KEY_HEADER_NAME} header.",
            headers={"WWW-Authenticate": API_KEY_HEADER_NAME},
        )

    # Every candidate is compared, with no early exit, so total time does not
    # depend on which key matched or how many were configured.
    matched = False
    for candidate in configured:
        if secrets.compare_digest(api_key, candidate):
            matched = True
    if not matched:
        raise UnauthorizedError(
            "Invalid API key.", headers={"WWW-Authenticate": API_KEY_HEADER_NAME}
        )


StateDep = Annotated[AppState, Depends(get_state)]
ConfigDep = Annotated[Config, Depends(get_config)]
MetricsDep = Annotated[ServiceMetrics, Depends(get_metrics)]
BatcherDep = Annotated[DynamicBatcher, Depends(get_batcher)]
ModelDep = Annotated[ModelBackend, Depends(get_model)]
AuthDep = Annotated[None, Depends(require_api_key)]
