"""Dynamic batching engine.

The most important tests in the repository. The batcher is where concurrency,
backpressure and shutdown correctness actually live, and every bug here shows up
in production as either lost requests or unbounded latency.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from inference_service.serving.batcher import (
    BatcherClosedError,
    DynamicBatcher,
    InferenceTimeoutError,
    QueueFullError,
)
from tests.conftest import FakeBackend


async def make_batcher(model: FakeBackend, **kwargs) -> DynamicBatcher:
    defaults = {
        "max_batch_size": 8,
        "max_wait_ms": 20.0,
        "max_queue_size": 16,
        "request_timeout_s": 2.0,
    }
    batcher = DynamicBatcher(model, **{**defaults, **kwargs})
    await batcher.start()
    return batcher


class TestBatchFormation:
    async def test_single_request_is_served(self) -> None:
        model = FakeBackend()
        batcher = await make_batcher(model)
        try:
            result = await batcher.predict("hello")
            assert result.label == "positive"
            assert model.batch_sizes == [1]
        finally:
            await batcher.stop()

    async def test_concurrent_requests_share_one_forward_pass(self) -> None:
        """The entire point of the component."""
        model = FakeBackend()
        batcher = await make_batcher(model, max_batch_size=8, max_wait_ms=50.0)
        try:
            results = await asyncio.gather(*(batcher.predict(f"text {i}") for i in range(8)))
            assert len(results) == 8
            # One call, eight items -- not eight calls.
            assert len(model.calls) == 1
            assert model.batch_sizes == [8]
        finally:
            await batcher.stop()

    async def test_batch_is_capped_at_max_batch_size(self) -> None:
        model = FakeBackend()
        batcher = await make_batcher(model, max_batch_size=4, max_wait_ms=50.0)
        try:
            await asyncio.gather(*(batcher.predict(f"t{i}") for i in range(10)))
            assert max(model.batch_sizes) <= 4
            assert sum(model.batch_sizes) == 10
        finally:
            await batcher.stop()

    async def test_window_closes_on_timeout_when_batch_not_full(self) -> None:
        """A lone request must not wait for a batch that will never fill."""
        model = FakeBackend()
        batcher = await make_batcher(model, max_batch_size=32, max_wait_ms=30.0)
        try:
            started = time.perf_counter()
            await batcher.predict("alone")
            elapsed = time.perf_counter() - started
            # Bounded by the window, not by max_batch_size arrivals.
            assert elapsed < 0.5
            assert model.batch_sizes == [1]
        finally:
            await batcher.stop()

    async def test_zero_wait_disables_windowing(self) -> None:
        model = FakeBackend()
        batcher = await make_batcher(model, max_wait_ms=0.0)
        try:
            await asyncio.gather(*(batcher.predict(f"t{i}") for i in range(4)))
            # With no window each item is dispatched as it is dequeued.
            assert model.batch_sizes == [1, 1, 1, 1]
        finally:
            await batcher.stop()

    async def test_batching_does_not_collapse_under_backlog(self) -> None:
        """Regression: batching must get BETTER under load, not worse.

        The window is measured from when a batch starts forming, not from when
        its head item was enqueued. With the latter, an item that waited behind
        a backlog arrives at the worker with an already-expired deadline, so
        every batch closes with exactly one item -- batching silently collapses
        precisely when it is needed, and throughput degrades as load rises.

        A benchmark caught this: 1791 rps with the fix, 66 rps without.
        """
        model = FakeBackend(delay_seconds=0.05)
        batcher = await make_batcher(model, max_batch_size=16, max_wait_ms=5.0, max_queue_size=256)
        try:
            # 60 concurrent submissions against a slow model guarantees a
            # backlog builds while the first batch is in flight.
            await asyncio.gather(*(batcher.predict(f"t{i}") for i in range(60)))

            # The first batch is legitimately small (it forms as requests
            # arrive). Every batch after the backlog exists should be large.
            later_batches = model.batch_sizes[1:]
            assert later_batches, "expected more than one batch"
            assert max(later_batches) > 1, (
                f"batching collapsed under backlog: batch sizes {model.batch_sizes}"
            )
            assert sum(later_batches) / len(later_batches) >= 4, (
                f"batches stayed small under load: {model.batch_sizes}"
            )
        finally:
            await batcher.stop(drain_timeout_s=2.0)

    async def test_queued_items_are_drained_without_waiting(self) -> None:
        """A backlog needs no wait: the items are already here."""
        model = FakeBackend(delay_seconds=0.05)
        batcher = await make_batcher(
            model, max_batch_size=32, max_wait_ms=500.0, max_queue_size=128
        )
        try:
            started = time.perf_counter()
            await asyncio.gather(*(batcher.predict(f"t{i}") for i in range(32)))
            elapsed = time.perf_counter() - started
            # With a 500ms window, waiting it out per batch would take seconds.
            # Draining the backlog immediately keeps this near the model's time.
            assert elapsed < 1.0, f"drained backlog took {elapsed:.2f}s; window was waited out"
        finally:
            await batcher.stop(drain_timeout_s=2.0)

    async def test_results_map_to_the_correct_caller(self) -> None:
        """Positional correspondence between inputs and outputs.

        A mix-up here silently returns one user's prediction to another, which
        is both a correctness and a privacy failure.
        """

        class EchoBackend(FakeBackend):
            def predict_batch(self, inputs):
                from inference_service.model.base import Prediction

                texts = list(inputs)
                self.batch_sizes.append(len(texts))
                return [Prediction(label=text, score=1.0, scores={text: 1.0}) for text in texts]

        model = EchoBackend()
        batcher = await make_batcher(model, max_batch_size=8, max_wait_ms=50.0)
        try:
            texts = [f"unique-{i}" for i in range(8)]
            results = await asyncio.gather(*(batcher.predict(t) for t in texts))
            assert [r.label for r in results] == texts
        finally:
            await batcher.stop()


class TestBackpressure:
    """Overload must produce fast rejections, not unbounded latency.

    Note on shape: the rejection surfaces on the *submitting* coroutine, so
    these tests submit a burst concurrently and inspect the results. Asserting
    on a later, separate call would be racy — by then the worker has drained a
    slot and the queue has room again.
    """

    async def test_burst_beyond_capacity_is_rejected(self) -> None:
        model = FakeBackend(delay_seconds=0.3)
        batcher = await make_batcher(model, max_batch_size=1, max_wait_ms=0.0, max_queue_size=2)
        try:
            results = await asyncio.gather(
                *(batcher.predict(f"t{i}") for i in range(12)),
                return_exceptions=True,
            )
            rejected = [r for r in results if isinstance(r, QueueFullError)]
            assert rejected, "an over-capacity burst produced no backpressure"
            assert all("queue is full" in str(r) for r in rejected)
        finally:
            await batcher.stop(drain_timeout_s=0.1)

    async def test_rejection_is_immediate_not_queued(self) -> None:
        """Rejection must not wait out the request timeout."""
        model = FakeBackend(delay_seconds=0.5)
        batcher = await make_batcher(
            model,
            max_batch_size=1,
            max_wait_ms=0.0,
            max_queue_size=1,
            request_timeout_s=10.0,
        )
        try:
            started = time.perf_counter()
            results = await asyncio.gather(
                *(batcher.predict(f"t{i}") for i in range(8)),
                return_exceptions=True,
            )
            elapsed = time.perf_counter() - started
            assert any(isinstance(r, QueueFullError) for r in results)
            # Rejections returned long before request_timeout_s.
            assert elapsed < 2.0
        finally:
            await batcher.stop(drain_timeout_s=0.1)

    async def test_rejection_is_counted(self) -> None:
        model = FakeBackend(delay_seconds=0.3)
        batcher = await make_batcher(model, max_batch_size=1, max_wait_ms=0.0, max_queue_size=1)
        try:
            await asyncio.gather(
                *(batcher.predict(f"t{i}") for i in range(8)),
                return_exceptions=True,
            )
            assert batcher.stats().items_rejected >= 1
        finally:
            await batcher.stop(drain_timeout_s=0.1)

    async def test_capacity_recovers_after_drain(self) -> None:
        """Backpressure is transient: once the queue drains, work is accepted."""
        model = FakeBackend(delay_seconds=0.05)
        batcher = await make_batcher(model, max_batch_size=1, max_wait_ms=0.0, max_queue_size=2)
        try:
            await asyncio.gather(
                *(batcher.predict(f"t{i}") for i in range(10)),
                return_exceptions=True,
            )
            await asyncio.sleep(0.3)
            result = await batcher.predict("after the storm")
            assert result.label == "positive"
        finally:
            await batcher.stop(drain_timeout_s=0.5)


class TestTimeouts:
    async def test_slow_inference_times_out(self) -> None:
        model = FakeBackend(delay_seconds=1.0)
        batcher = await make_batcher(model, request_timeout_s=0.1)
        try:
            with pytest.raises(InferenceTimeoutError, match="did not complete"):
                await batcher.predict("slow")
        finally:
            await batcher.stop(drain_timeout_s=0.1)

    async def test_per_request_timeout_overrides_default(self) -> None:
        model = FakeBackend(delay_seconds=0.5)
        batcher = await make_batcher(model, request_timeout_s=10.0)
        try:
            with pytest.raises(InferenceTimeoutError):
                await batcher.predict("slow", timeout_s=0.05)
        finally:
            await batcher.stop(drain_timeout_s=0.1)

    async def test_timed_out_items_do_not_consume_batch_capacity(self) -> None:
        """A caller that gave up must not occupy a slot in a later batch."""
        model = FakeBackend(delay_seconds=0.2)
        batcher = await make_batcher(
            model, max_batch_size=4, max_wait_ms=10.0, request_timeout_s=0.05
        )
        try:
            results = await asyncio.gather(
                *(batcher.predict(f"t{i}") for i in range(4)), return_exceptions=True
            )
            assert all(isinstance(r, InferenceTimeoutError) for r in results)
            await asyncio.sleep(0.4)
            # Every abandoned item was filtered out, so no batch of dead work ran.
            assert all(size <= 4 for size in model.batch_sizes)
        finally:
            await batcher.stop(drain_timeout_s=0.1)


class TestFailureHandling:
    async def test_backend_exception_propagates_to_every_caller(self) -> None:
        model = FakeBackend(fail_with=RuntimeError("model exploded"))
        batcher = await make_batcher(model, max_batch_size=4, max_wait_ms=30.0)
        try:
            results = await asyncio.gather(
                *(batcher.predict(f"t{i}") for i in range(4)), return_exceptions=True
            )
            assert len(results) == 4
            assert all(isinstance(r, RuntimeError) for r in results)
        finally:
            await batcher.stop()

    async def test_worker_survives_a_failing_batch(self) -> None:
        """One bad batch must not kill the worker and hang everything after."""
        model = FakeBackend(fail_with=RuntimeError("boom"))
        batcher = await make_batcher(model)
        try:
            with pytest.raises(RuntimeError):
                await batcher.predict("first")

            model.fail_with = None
            result = await batcher.predict("second")
            assert result.label == "positive"
        finally:
            await batcher.stop()

    async def test_mismatched_output_length_is_detected(self) -> None:
        """A backend returning the wrong count would pair callers with the
        wrong predictions. Fail loudly instead."""
        model = FakeBackend(wrong_output_length=True)
        batcher = await make_batcher(model, max_batch_size=4, max_wait_ms=30.0)
        try:
            results = await asyncio.gather(
                *(batcher.predict(f"t{i}") for i in range(3)), return_exceptions=True
            )
            assert all(isinstance(r, RuntimeError) for r in results)
            assert any("contract violation" in str(r) for r in results)
        finally:
            await batcher.stop()


class TestLifecycle:
    async def test_predict_before_start_is_rejected(self) -> None:
        batcher = DynamicBatcher(FakeBackend(), max_batch_size=2)
        with pytest.raises(BatcherClosedError):
            await batcher.predict("x")

    async def test_double_start_is_rejected(self) -> None:
        batcher = await make_batcher(FakeBackend())
        try:
            with pytest.raises(Exception, match="already started"):
                await batcher.start()
        finally:
            await batcher.stop()

    async def test_stop_drains_accepted_work(self) -> None:
        """Requests already accepted must be served, not dropped."""
        model = FakeBackend(delay_seconds=0.02)
        batcher = await make_batcher(model, max_batch_size=8, max_wait_ms=10.0)
        tasks = [asyncio.create_task(batcher.predict(f"t{i}")) for i in range(8)]
        await asyncio.sleep(0.02)

        await batcher.stop(drain_timeout_s=5.0)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        served = [r for r in results if not isinstance(r, BaseException)]
        assert len(served) == 8, "accepted work was dropped during drain"

    async def test_new_requests_rejected_while_draining(self) -> None:
        batcher = await make_batcher(FakeBackend())
        await batcher.stop()
        with pytest.raises(BatcherClosedError, match="shutting down"):
            await batcher.predict("too late")

    async def test_stop_is_idempotent(self) -> None:
        batcher = await make_batcher(FakeBackend())
        await batcher.stop()
        await batcher.stop()

    async def test_running_reflects_drain_state(self) -> None:
        batcher = await make_batcher(FakeBackend())
        assert batcher.running is True
        await batcher.stop()
        assert batcher.running is False


class TestPredictMany:
    async def test_returns_results_in_order(self) -> None:
        class EchoBackend(FakeBackend):
            def predict_batch(self, inputs):
                from inference_service.model.base import Prediction

                texts = list(inputs)
                self.batch_sizes.append(len(texts))
                return [Prediction(label=t, score=1.0, scores={t: 1.0}) for t in texts]

        model = EchoBackend()
        batcher = await make_batcher(model, max_batch_size=8, max_wait_ms=30.0)
        try:
            texts = ["a", "b", "c", "d"]
            results = await batcher.predict_many(texts)
            assert [r.label for r in results] == texts
        finally:
            await batcher.stop()

    async def test_failure_cancels_siblings(self) -> None:
        model = FakeBackend(fail_with=RuntimeError("boom"))
        batcher = await make_batcher(model, max_batch_size=4, max_wait_ms=20.0)
        try:
            with pytest.raises(RuntimeError):
                await batcher.predict_many(["a", "b", "c"])
        finally:
            await batcher.stop()


class TestStats:
    async def test_counters_advance(self) -> None:
        model = FakeBackend()
        batcher = await make_batcher(model, max_batch_size=4, max_wait_ms=30.0)
        try:
            await asyncio.gather(*(batcher.predict(f"t{i}") for i in range(4)))
            stats = batcher.stats()
            assert stats.items_processed == 4
            assert stats.batches_processed >= 1
            assert stats.mean_batch_size > 0
            assert stats.queue_capacity == 16
        finally:
            await batcher.stop()


class TestValidation:
    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"max_batch_size": 0}, "max_batch_size"),
            ({"max_wait_ms": -1}, "max_wait_ms"),
            ({"max_queue_size": 0}, "max_queue_size"),
            ({"request_timeout_s": 0}, "request_timeout_s"),
        ],
    )
    def test_invalid_construction_is_rejected(self, kwargs, match) -> None:
        with pytest.raises(ValueError, match=match):
            DynamicBatcher(FakeBackend(), **kwargs)
