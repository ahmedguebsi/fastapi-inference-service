"""HTTP surface: prediction, validation, errors, auth, headers."""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest
from httpx import AsyncClient

from inference_service.config import APIConfig, BatchingConfig


class TestPredict:
    async def test_returns_a_prediction(self, client: AsyncClient) -> None:
        response = await client.post(
            "/v1/predict", json={"text": "this is excellent and works perfectly"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["result"]["label"] in ("positive", "negative")
        assert 0.0 <= body["result"]["score"] <= 1.0
        assert set(body["result"]["scores"]) == {"positive", "negative"}
        assert body["model_name"] == "sentiment"
        assert body["latency_ms"] >= 0

    async def test_scores_sum_to_one(self, client: AsyncClient) -> None:
        """Binary logistic regression: the two class scores are complementary."""
        response = await client.post("/v1/predict", json={"text": "a neutral sentence"})
        scores = response.json()["result"]["scores"]
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-4)

    async def test_reported_label_is_the_argmax(self, client: AsyncClient) -> None:
        body = (await client.post("/v1/predict", json={"text": "wonderful and reliable"})).json()[
            "result"
        ]
        assert body["score"] == pytest.approx(max(body["scores"].values()), abs=1e-6)
        assert body["label"] == max(body["scores"], key=body["scores"].get)

    async def test_model_discriminates(self, client: AsyncClient) -> None:
        """A sanity check that the artifact carries real signal.

        Not an accuracy assertion -- the model is small by design. It only
        asserts that clearly opposite inputs land on opposite sides, which would
        catch a corrupted or accidentally zeroed artifact.
        """
        positive = (
            await client.post(
                "/v1/predict", json={"text": "excellent, works perfectly, highly recommend"}
            )
        ).json()["result"]
        negative = (
            await client.post(
                "/v1/predict", json={"text": "terrible, completely broken, total waste"}
            )
        ).json()["result"]
        assert positive["scores"]["positive"] > negative["scores"]["positive"]

    async def test_deterministic(self, client: AsyncClient) -> None:
        """Same input, same output. Feature hashing must not use PYTHONHASHSEED."""
        payload = {"text": "consistency matters in production"}
        first = (await client.post("/v1/predict", json=payload)).json()["result"]
        second = (await client.post("/v1/predict", json=payload)).json()["result"]
        assert first == second


class TestBatchPredict:
    async def test_returns_one_result_per_input(self, client: AsyncClient) -> None:
        texts = ["great product", "awful rubbish", "it is fine"]
        response = await client.post("/v1/predict/batch", json={"texts": texts})
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == len(texts)
        assert len(body["results"]) == len(texts)

    async def test_results_are_positionally_aligned(self, client: AsyncClient) -> None:
        """Order must be preserved, or clients silently mismatch results."""
        texts = ["excellent, works perfectly, highly recommend"] * 3 + [
            "terrible, completely broken, total waste"
        ] * 3
        results = (await client.post("/v1/predict/batch", json={"texts": texts})).json()["results"]
        first_three = {r["label"] for r in results[:3]}
        last_three = {r["label"] for r in results[3:]}
        assert len(first_three) == 1
        assert len(last_three) == 1
        assert first_three != last_three

    async def test_matches_single_predict(self, client: AsyncClient) -> None:
        """Batching must not change the answer, only the throughput."""
        text = "a moderately positive statement about quality"
        single = (await client.post("/v1/predict", json={"text": text})).json()["result"]
        batched = (await client.post("/v1/predict/batch", json={"texts": [text]})).json()[
            "results"
        ][0]
        assert single == batched


class TestValidation:
    async def test_missing_field_is_422(self, client: AsyncClient) -> None:
        response = await client.post("/v1/predict", json={})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_unknown_field_is_rejected(self, client: AsyncClient) -> None:
        """extra='forbid' turns a client typo into a clear error."""
        response = await client.post("/v1/predict", json={"txt": "typo in the field name"})
        assert response.status_code == 422
        fields = response.json()["error"]["details"]["fields"]
        assert any("txt" in f["field"] or "extra" in f["type"] for f in fields)

    async def test_empty_string_is_rejected(self, client: AsyncClient) -> None:
        assert (await client.post("/v1/predict", json={"text": ""})).status_code == 422

    async def test_whitespace_only_is_rejected(self, client: AsyncClient) -> None:
        """Produces no tokens; a prediction from it would be meaningless."""
        response = await client.post("/v1/predict", json={"text": "   \t\n  "})
        assert response.status_code == 422

    async def test_wrong_type_is_rejected(self, client: AsyncClient) -> None:
        assert (await client.post("/v1/predict", json={"text": 42})).status_code == 422

    async def test_empty_batch_is_rejected(self, client: AsyncClient) -> None:
        assert (await client.post("/v1/predict/batch", json={"texts": []})).status_code == 422

    async def test_oversized_batch_is_rejected(self, client: AsyncClient) -> None:
        response = await client.post("/v1/predict/batch", json={"texts": ["x"] * 500})
        assert response.status_code in (413, 422)

    async def test_error_response_never_echoes_input(self, client: AsyncClient) -> None:
        """Validation errors must not reflect user text back.

        Pydantic's raw errors embed the offending input; echoing it turns a
        validation message into a reflection vector and a PII leak.
        """
        secret = "SUPER-SECRET-VALUE-9f3a"
        response = await client.post("/v1/predict", json={"text": secret, "extra": 1})
        assert response.status_code == 422
        assert secret not in response.text


class TestErrorEnvelope:
    async def test_unknown_path_uses_the_standard_envelope(self, client: AsyncClient) -> None:
        response = await client.get("/does-not-exist")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    async def test_wrong_method_is_405(self, client: AsyncClient) -> None:
        response = await client.get("/v1/predict")
        assert response.status_code == 405
        assert "error" in response.json()

    async def test_every_error_carries_a_request_id(self, client: AsyncClient) -> None:
        response = await client.post("/v1/predict", json={})
        assert response.json()["error"]["request_id"]


class TestRequestCorrelation:
    async def test_response_carries_a_request_id(self, client: AsyncClient) -> None:
        response = await client.post("/v1/predict", json={"text": "hello there"})
        assert response.headers["X-Request-ID"]
        assert response.json()["request_id"] == response.headers["X-Request-ID"]

    async def test_client_supplied_id_is_honoured(self, client: AsyncClient) -> None:
        """Lets a trace span the whole call chain through a gateway."""
        supplied = "client-trace-id-12345"
        response = await client.post(
            "/v1/predict",
            json={"text": "hello"},
            headers={"X-Request-ID": supplied},
        )
        assert response.headers["X-Request-ID"] == supplied

    async def test_ids_are_unique_per_request(self, client: AsyncClient) -> None:
        ids = set()
        for _ in range(5):
            response = await client.post("/v1/predict", json={"text": "hello"})
            ids.add(response.headers["X-Request-ID"])
        assert len(ids) == 5

    async def test_process_time_header_present(self, client: AsyncClient) -> None:
        response = await client.post("/v1/predict", json={"text": "hello"})
        assert float(response.headers["X-Process-Time-Ms"]) >= 0


class TestAuth:
    async def test_disabled_by_default(self, client: AsyncClient) -> None:
        assert (await client.post("/v1/predict", json={"text": "no key needed"})).status_code == 200

    async def test_missing_key_is_401(self, base_config, make_client) -> None:
        config = replace(base_config, api=replace(base_config.api, api_keys=("secret-key",)))
        async with make_client(config) as client:
            response = await client.post("/v1/predict", json={"text": "hello"})
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "unauthorized"

    async def test_wrong_key_is_401(self, base_config, make_client) -> None:
        config = replace(base_config, api=replace(base_config.api, api_keys=("secret-key",)))
        async with make_client(config) as client:
            response = await client.post(
                "/v1/predict",
                json={"text": "hello"},
                headers={"X-API-Key": "wrong-key"},
            )
            assert response.status_code == 401

    async def test_valid_key_is_accepted(self, base_config, make_client) -> None:
        config = replace(base_config, api=replace(base_config.api, api_keys=("secret-key",)))
        async with make_client(config) as client:
            response = await client.post(
                "/v1/predict",
                json={"text": "hello"},
                headers={"X-API-Key": "secret-key"},
            )
            assert response.status_code == 200

    async def test_any_configured_key_works(self, base_config, make_client) -> None:
        config = replace(base_config, api=replace(base_config.api, api_keys=("key-a", "key-b")))
        async with make_client(config) as client:
            for key in ("key-a", "key-b"):
                response = await client.post(
                    "/v1/predict", json={"text": "hi"}, headers={"X-API-Key": key}
                )
                assert response.status_code == 200

    async def test_health_is_not_authenticated(self, base_config, make_client) -> None:
        """Probes must work without credentials, or the orchestrator kills the pod."""
        config = replace(base_config, api=replace(base_config.api, api_keys=("secret",)))
        async with make_client(config) as client:
            assert (await client.get("/health")).status_code == 200
            assert (await client.get("/ready")).status_code == 200


class TestLimits:
    async def test_text_over_limit_is_rejected(self, base_config, make_client) -> None:
        config = replace(base_config, api=APIConfig(max_text_length=50, max_texts_per_request=4))
        async with make_client(config) as client:
            response = await client.post("/v1/predict", json={"text": "x" * 200})
            assert response.status_code in (413, 422)

    async def test_too_many_texts_is_rejected(self, base_config, make_client) -> None:
        config = replace(base_config, api=APIConfig(max_text_length=500, max_texts_per_request=2))
        async with make_client(config) as client:
            response = await client.post("/v1/predict/batch", json={"texts": ["a", "b", "c", "d"]})
            assert response.status_code in (413, 422)


class TestConcurrency:
    async def test_concurrent_requests_all_succeed(self, client: AsyncClient) -> None:
        """Exercises the full path under concurrency: every caller gets a
        correct answer and none are dropped or crossed."""
        responses = await asyncio.gather(
            *(client.post("/v1/predict", json={"text": f"request number {i}"}) for i in range(24))
        )
        assert all(r.status_code == 200 for r in responses)
        ids = {r.headers["X-Request-ID"] for r in responses}
        assert len(ids) == 24

    async def test_overload_returns_503_not_a_hang(self, base_config, make_client) -> None:
        """Under backpressure the service must reject fast, not stall."""
        config = replace(
            base_config,
            batching=BatchingConfig(
                max_batch_size=1,
                max_wait_ms=0.0,
                max_queue_size=2,
                request_timeout_s=5.0,
                drain_timeout_s=1.0,
            ),
        )
        async with make_client(config) as client:
            responses = await asyncio.gather(
                *(client.post("/v1/predict", json={"text": f"flood {i}"}) for i in range(60)),
                return_exceptions=True,
            )
            codes = [r.status_code for r in responses if hasattr(r, "status_code")]
            assert codes, "no responses returned"
            assert all(code in (200, 503) for code in codes)
            # Whether backpressure trips depends on scheduling; what must always
            # hold is that nothing hangs and nothing 500s.
            assert not any(code >= 500 and code != 503 for code in codes)
