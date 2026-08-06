"""Shared fixtures.

Every test builds its own application via ``create_app`` with an isolated
:class:`~prometheus_client.CollectorRegistry`. That is why the factory pattern
exists: a module-level ``app`` would share one registry across the suite and
every test after the first would fail with a duplicate-timeseries error.

``httpx.ASGITransport`` drives the app in-process — no socket, no port, no
sleeping to wait for a server to boot. ``LifespanManager`` is hand-rolled below
because the model load and batcher startup live in the lifespan, and a test that
skips it would be testing an app that can never serve a request.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client import CollectorRegistry

from inference_service.config import (
    APIConfig,
    BatchingConfig,
    Config,
    LoggingConfig,
    ModelConfig,
    ObservabilityConfig,
    ServerConfig,
)
from inference_service.main import create_app
from inference_service.metrics import ServiceMetrics
from inference_service.model.base import ModelBackend, ModelInfo, Prediction

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_PATH = REPO_ROOT / "models" / "sentiment-v1.json"


# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------
class FakeBackend(ModelBackend):
    """Controllable backend, so serving behaviour can be tested in isolation.

    The real model is fast and always succeeds, which makes it useless for
    testing timeouts, backpressure and inference failures. This one can be told
    to be slow, to fail, or to violate the batch-length contract.
    """

    BACKEND_NAME = "fake"

    def __init__(
        self,
        *,
        delay_seconds: float = 0.0,
        fail_with: Exception | None = None,
        label: str = "positive",
        wrong_output_length: bool = False,
    ) -> None:
        self.delay_seconds = delay_seconds
        self.fail_with = fail_with
        self.label = label
        self.wrong_output_length = wrong_output_length
        self.calls: list[list[str]] = []
        self.batch_sizes: list[int] = []
        self.closed = False
        self.warmed_up = False
        self._loaded = False

    @property
    def info(self) -> ModelInfo:
        return ModelInfo(
            name="fake",
            version="0.0.1",
            task="text-classification",
            labels=("negative", "positive"),
            backend=self.BACKEND_NAME,
            artifact_sha256="0" * 64,
            parameters=1,
        )

    def load(self) -> None:
        self._loaded = True

    def warmup(self) -> None:
        self.warmed_up = True

    def predict_batch(self, inputs: Any) -> list[Prediction]:
        texts = list(inputs)
        self.calls.append(texts)
        self.batch_sizes.append(len(texts))

        if self.delay_seconds:
            import time

            time.sleep(self.delay_seconds)
        if self.fail_with is not None:
            raise self.fail_with
        if self.wrong_output_length:
            return []

        return [
            Prediction(
                label=self.label,
                score=0.9,
                scores={"negative": 0.1, "positive": 0.9},
            )
            for _ in texts
        ]

    def close(self) -> None:
        self.closed = True


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@pytest.fixture
def base_config() -> Config:
    """Test-shaped config: real artifact, no auth, batching windows tightened.

    ``max_wait_ms`` is small so tests do not spend real time waiting for a batch
    window that exists to serve production throughput, not test runtime.
    """
    return Config(
        service_name="inference-service-test",
        environment="test",
        # prestop_delay_s=0: that delay exists to let a load balancer notice
        # readiness flipping. There is no LB in tests, and paying it on every
        # teardown added ~5s per test (~3 minutes across the suite).
        server=ServerConfig(prestop_delay_s=0.0),
        model=ModelConfig(artifact_path=str(ARTIFACT_PATH), warmup=True),
        batching=BatchingConfig(
            max_batch_size=8,
            max_wait_ms=5.0,
            # Comfortably above the concurrency the API tests generate, so
            # backpressure only trips in the tests that deliberately induce it.
            max_queue_size=128,
            request_timeout_s=5.0,
            drain_timeout_s=2.0,
        ),
        api=APIConfig(max_texts_per_request=8, max_text_length=500),
        logging=LoggingConfig(level="WARNING", format="text", access_log=False),
        observability=ObservabilityConfig(),
    ).validate()


@pytest.fixture
def metrics() -> ServiceMetrics:
    """Metrics bound to a private registry, so tests never collide."""
    return ServiceMetrics(ObservabilityConfig(), registry=CollectorRegistry())


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class LifespanManager:
    """Run an ASGI app's lifespan around a block.

    httpx does not trigger lifespan events, and this application does all of its
    real work there — model load, warmup, batcher start. Without this the app
    under test would have no model and every request would 503, so the tests
    would be exercising a service that can never actually serve.

    Uses ``router.lifespan_context``, the same coroutine Starlette drives from
    its ASGI lifespan handler, so tests run the identical startup and shutdown
    path production does — including the graceful drain.
    """

    def __init__(self, app: Any) -> None:
        self._app = app
        self._context: Any = None

    async def __aenter__(self) -> Any:
        self._context = self._app.router.lifespan_context(self._app)
        await self._context.__aenter__()
        return self._app

    async def __aexit__(self, *exc_info: object) -> None:
        if self._context is not None:
            await self._context.__aexit__(None, None, None)
            self._context = None


@pytest.fixture
async def client(base_config: Config, metrics: ServiceMetrics) -> AsyncIterator[AsyncClient]:
    """An HTTP client bound to a fully started app."""
    app = create_app(base_config, metrics=metrics)
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            yield http_client


@pytest.fixture
def make_client(metrics: ServiceMetrics):
    """Factory for clients with modified config.

    Returns an async context manager so tests can vary auth, batching limits or
    the model path without duplicating the wiring above.
    """

    def _factory(config: Config, *, own_metrics: ServiceMetrics | None = None):
        class _Ctx:
            async def __aenter__(self) -> AsyncClient:
                self.app = create_app(config, metrics=own_metrics or metrics)
                self.lifespan = LifespanManager(self.app)
                await self.lifespan.__aenter__()
                self.transport = ASGITransport(app=self.app)
                self.client = AsyncClient(transport=self.transport, base_url="http://test")
                await self.client.__aenter__()
                return self.client

            async def __aexit__(self, *exc: object) -> None:
                await self.client.__aexit__(*exc)
                await self.lifespan.__aexit__(*exc)

        return _Ctx()

    return _factory


@pytest.fixture
def config_with(base_config: Config):
    """Shorthand for producing a variant of the base config."""

    def _with(**changes: Any) -> Config:
        return replace(base_config, **changes).validate()

    return _with


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------
@pytest.fixture
def artifact_payload() -> dict[str, Any]:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def write_artifact(tmp_path: Path):
    """Write a (possibly malformed) artifact and return its path."""

    def _write(payload: Any, name: str = "model.json") -> Path:
        path = tmp_path / name
        path.write_text(
            payload if isinstance(payload, str) else json.dumps(payload),
            encoding="utf-8",
        )
        return path

    return _write


@pytest.fixture(scope="session")
def real_artifact_exists() -> Iterator[None]:
    if not ARTIFACT_PATH.is_file():
        pytest.skip(
            f"model artifact missing at {ARTIFACT_PATH}; run: python scripts/train_model.py"
        )
    yield
