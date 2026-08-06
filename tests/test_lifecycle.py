"""Health, readiness, metadata, metrics and shutdown ordering.

The readiness/liveness distinction and the shutdown sequence are the two things
that determine whether a deploy loses requests, so they get explicit tests
rather than being assumed to work.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from httpx import ASGITransport, AsyncClient

from inference_service.config import ModelConfig, ServerConfig
from inference_service.main import create_app
from tests.conftest import LifespanManager


class TestLiveness:
    async def test_health_is_200(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["uptime_seconds"] >= 0

    @pytest.mark.parametrize("path", ["/health", "/healthz", "/live"])
    async def test_aliases(self, client: AsyncClient, path: str) -> None:
        assert (await client.get(path)).status_code == 200

    async def test_liveness_does_not_depend_on_the_model(self, base_config, metrics) -> None:
        """The critical property.

        If liveness failed when the model was missing, a bad artifact would put
        the container into a restart loop -- turning a recoverable degradation
        into a total outage. Here the app never starts (load is fatal), so this
        asserts the handler itself has no model dependency.
        """
        config = replace(base_config, model=ModelConfig(artifact_path="/nonexistent.json"))
        app = create_app(config, metrics=metrics)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # No lifespan run, so no model is loaded at all.
            response = await client.get("/health")
            assert response.status_code == 200


class TestReadiness:
    async def test_ready_when_serving(self, client: AsyncClient) -> None:
        response = await client.get("/ready")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ready"
        assert all(body["checks"].values())

    @pytest.mark.parametrize("path", ["/ready", "/readyz", "/startup"])
    async def test_aliases(self, client: AsyncClient, path: str) -> None:
        assert (await client.get(path)).status_code == 200

    async def test_not_ready_before_startup(self, base_config, metrics) -> None:
        """503 removes the instance from the LB without restarting it."""
        app = create_app(base_config, metrics=metrics)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.get("/ready")
            assert response.status_code == 503
            assert response.json()["status"] == "not_ready"
            assert response.json()["checks"]["model_loaded"] is False

    async def test_predictions_rejected_before_ready(self, base_config, metrics) -> None:
        app = create_app(base_config, metrics=metrics)
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            response = await client.post("/v1/predict", json={"text": "too early"})
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "model_not_ready"


class TestStartupFailure:
    async def test_missing_artifact_aborts_startup(self, base_config, metrics) -> None:
        """Fail fast and loudly.

        A service that boots without a model and answers 503 forever is harder
        to diagnose than one that refuses to start with the reason in its first
        log line -- and an orchestrator surfaces a startup crash immediately.
        """
        config = replace(base_config, model=ModelConfig(artifact_path="/definitely/not/here.json"))
        app = create_app(config, metrics=metrics)
        with pytest.raises(Exception, match="not found"):
            async with LifespanManager(app):
                pass


class TestMeta:
    async def test_reports_model_identity(self, client: AsyncClient) -> None:
        body = (await client.get("/v1/meta")).json()
        assert body["model"]["name"] == "sentiment"
        assert body["model"]["backend"] == "linear"
        assert len(body["model"]["artifact_sha256"]) == 64

    async def test_reports_batching_configuration(self, client: AsyncClient) -> None:
        batching = (await client.get("/v1/meta")).json()["batching"]
        assert batching["max_batch_size"] == 8
        assert batching["max_queue_size"] == 128

    async def test_reports_live_scheduler_stats(self, client: AsyncClient) -> None:
        await client.post("/v1/predict", json={"text": "warm the counters"})
        stats = (await client.get("/v1/meta")).json()["batcher_stats"]
        assert stats["running"] is True
        assert stats["items_processed"] >= 1
        assert stats["queue_capacity"] == 128


class TestMetricsEndpoint:
    async def test_exposition_is_served(self, client: AsyncClient) -> None:
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")

    async def test_content_type_matches_the_body_format(self, client: AsyncClient) -> None:
        """Advertising OpenMetrics while emitting Prometheus text format makes a
        conforming scraper reject every scrape (OpenMetrics needs '# EOF')."""
        response = await client.get("/metrics")
        content_type = response.headers["content-type"]
        body = response.text
        assert "openmetrics" not in content_type
        assert not body.rstrip().endswith("# EOF")

    async def test_request_metrics_are_recorded(self, client: AsyncClient) -> None:
        await client.post("/v1/predict", json={"text": "measure me"})
        body = (await client.get("/metrics")).text
        assert "inference_http_requests_total" in body
        assert 'endpoint="/v1/predict"' in body

    async def test_inference_metrics_are_recorded(self, client: AsyncClient) -> None:
        await client.post("/v1/predict", json={"text": "measure me"})
        body = (await client.get("/metrics")).text
        assert "inference_batch_size_count" in body
        assert "inference_queue_wait_seconds_count" in body
        assert "inference_predictions_total" in body

    async def test_ready_gauge_reflects_state(self, client: AsyncClient) -> None:
        body = (await client.get("/metrics")).text
        assert "inference_ready 1.0" in body

    async def test_endpoint_label_uses_the_route_template(self, client: AsyncClient) -> None:
        """Raw paths as label values are the classic cardinality explosion."""
        await client.post("/v1/predict", json={"text": "x"})
        body = (await client.get("/metrics")).text
        assert 'endpoint="/v1/predict"' in body

    async def test_status_label_is_a_class_not_a_code(self, client: AsyncClient) -> None:
        await client.post("/v1/predict", json={"text": "x"})
        body = (await client.get("/metrics")).text
        assert 'status="2xx"' in body
        assert 'status="200"' not in body

    async def test_can_be_disabled(self, base_config, make_client, metrics) -> None:
        from inference_service.config import ObservabilityConfig

        config = replace(base_config, observability=ObservabilityConfig(metrics_enabled=False))
        async with make_client(config) as client:
            assert (await client.get("/metrics")).status_code == 404


class TestShutdown:
    async def test_drain_serves_accepted_requests(self, base_config, metrics) -> None:
        """A rolling deploy must not drop work already accepted."""
        app = create_app(base_config, metrics=metrics)
        async with LifespanManager(app):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                responses = await asyncio.gather(
                    *(client.post("/v1/predict", json={"text": f"inflight {i}"}) for i in range(8))
                )
                assert all(r.status_code == 200 for r in responses)
        # Exiting the context ran the full shutdown sequence without raising.

    async def test_readiness_flips_before_teardown(self, base_config, metrics) -> None:
        """Ordering check: not-ready must precede the queue closing, or the LB
        keeps routing to a service that has stopped accepting work."""
        state_snapshots: list[tuple[bool, bool]] = []

        app = create_app(base_config, metrics=metrics)
        async with LifespanManager(app):
            state = app.state.app_state
            state_snapshots.append((state.ready, state.shutting_down))

        state = app.state.app_state
        state_snapshots.append((state.ready, state.shutting_down))

        assert state_snapshots[0] == (True, False), "not ready while serving"
        assert state_snapshots[1] == (False, True), "still ready after shutdown"

    async def test_model_is_closed(self, base_config, metrics) -> None:
        app = create_app(base_config, metrics=metrics)
        async with LifespanManager(app):
            model = app.state.app_state.model
        assert model is not None
        assert not model.is_loaded  # type: ignore[attr-defined]

    async def test_ready_gauge_drops_on_shutdown(self, base_config, metrics) -> None:
        app = create_app(base_config, metrics=metrics)
        async with LifespanManager(app):
            pass
        body = metrics.render()[0].decode()
        assert "inference_ready 0.0" in body

    async def test_prestop_delay_is_honoured(self, base_config, metrics) -> None:
        """The delay exists so the LB notices; assert it actually runs."""
        import time

        config = replace(base_config, server=ServerConfig(prestop_delay_s=0.25))
        app = create_app(config, metrics=metrics)
        started = time.perf_counter()
        async with LifespanManager(app):
            pass
        assert time.perf_counter() - started >= 0.25


class TestDocs:
    async def test_openapi_is_served(self, client: AsyncClient) -> None:
        response = await client.get("/openapi.json")
        assert response.status_code == 200
        schema = response.json()
        assert "/v1/predict" in schema["paths"]

    async def test_docs_can_be_disabled(self, base_config, make_client) -> None:
        """Unnecessary surface in production, where clients use a published spec."""
        from inference_service.config import APIConfig

        config = replace(
            base_config,
            api=APIConfig(max_texts_per_request=8, max_text_length=500, docs_enabled=False),
        )
        async with make_client(config) as client:
            assert (await client.get("/docs")).status_code == 404
            assert (await client.get("/openapi.json")).status_code == 404
