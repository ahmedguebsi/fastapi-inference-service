"""Dynamic request batching.

The single most important optimisation in model serving, and the thing that
separates an inference service from a CRUD API.

## The problem

Model inference has a large fixed cost per call — kernel launch, memory
transfer, framework overhead — and a comparatively small marginal cost per item.
Running one item per forward pass wastes most of that fixed cost. Batching 32
items into one pass is frequently 10-20x the throughput of 32 separate passes on
the same hardware.

But the requests arrive one at a time, from independent clients, and none of them
want to wait.

## The mechanism

Requests do not call the model. They enqueue an item carrying an
:class:`asyncio.Future` and await it. A single background worker drains the
queue: it takes the first item, then keeps collecting for up to
``max_wait_ms`` or until it has ``max_batch_size`` items, whichever comes
first. It runs one ``predict_batch`` call and resolves every future.

That window is the entire trade: latency for throughput, bounded and
configurable. Under low load the queue is empty, the wait expires with a single
item, and no latency worth measuring is added. Under load, items are already
queued and get drained into the batch immediately — so throughput climbs without
adding any wait at all. The system adapts without a tuning knob changing.

The window is measured from when a batch *starts forming*, not from when its
head item was enqueued. See :meth:`DynamicBatcher._collect_batch` — that
distinction is what stops batching from collapsing under load.

## Why the model runs in a thread

``predict_batch`` is synchronous and CPU-bound. Calling it directly on the event
loop would block every other request — including health checks — for its whole
duration. ``asyncio.to_thread`` moves it off the loop.

For a pure-Python backend the GIL means this yields no parallelism, only
responsiveness. For ONNX Runtime or PyTorch it yields real parallelism, because
those release the GIL inside their native inference calls. The design is correct
for both; only the magnitude of the benefit changes.

## Backpressure

The queue is bounded. When it is full the service returns 503 immediately rather
than accepting work it cannot finish. An unbounded queue under sustained
overload does not increase throughput — it converts a fast, honest rejection
into unbounded latency growth, memory exhaustion, and eventually an OOM kill
that takes the in-flight work with it.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ..logging_setup import get_logger
from ..model.base import ModelBackend, Prediction

if TYPE_CHECKING:
    from ..metrics import ServiceMetrics

logger = get_logger("batcher")


class BatcherError(RuntimeError):
    """Base class for batching failures."""


class QueueFullError(BatcherError):
    """The queue is at capacity. Surfaced to clients as 503 with Retry-After."""


class BatcherClosedError(BatcherError):
    """The batcher is shutting down and is not accepting new work."""


class InferenceTimeoutError(BatcherError):
    """A request waited longer than its budget. Surfaced as 504."""


@dataclass(slots=True)
class _BatchItem:
    """One queued request awaiting inference."""

    text: str
    future: asyncio.Future[Prediction]
    enqueued_at: float = field(default_factory=time.perf_counter)

    @property
    def queue_time_seconds(self) -> float:
        return time.perf_counter() - self.enqueued_at


@dataclass(frozen=True, slots=True)
class BatcherStats:
    """Point-in-time view, exposed on /v1/meta for debugging."""

    queue_depth: int
    queue_capacity: int
    batches_processed: int
    items_processed: int
    items_rejected: int
    items_timed_out: int
    running: bool

    @property
    def mean_batch_size(self) -> float:
        if self.batches_processed == 0:
            return 0.0
        return round(self.items_processed / self.batches_processed, 3)


class DynamicBatcher:
    """Collects concurrent requests into batches and resolves them individually."""

    def __init__(
        self,
        model: ModelBackend,
        *,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
        max_queue_size: int = 512,
        request_timeout_s: float = 10.0,
        metrics: ServiceMetrics | None = None,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be >= 1")
        if max_wait_ms < 0:
            raise ValueError("max_wait_ms must be >= 0")
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be >= 1")
        if request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be > 0")

        self._model = model
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_ms / 1000.0
        self._max_queue_size = max_queue_size
        self._request_timeout_s = request_timeout_s
        self._metrics = metrics

        self._queue: asyncio.Queue[_BatchItem] = asyncio.Queue(maxsize=max_queue_size)
        self._worker: asyncio.Task[None] | None = None
        self._running = False
        self._draining = False

        self._batches_processed = 0
        self._items_processed = 0
        self._items_rejected = 0
        self._items_timed_out = 0

    # -- lifecycle ----------------------------------------------------------
    async def start(self) -> None:
        if self._running:
            raise BatcherError("batcher already started")
        self._running = True
        self._draining = False
        self._worker = asyncio.create_task(self._run(), name="inference-batcher")
        logger.info(
            "batcher started",
            extra={
                "max_batch_size": self._max_batch_size,
                "max_wait_ms": self._max_wait_s * 1000,
                "max_queue_size": self._max_queue_size,
                "request_timeout_s": self._request_timeout_s,
            },
        )

    async def stop(self, drain_timeout_s: float = 30.0) -> None:
        """Stop accepting work, finish what is queued, then exit.

        Order matters. ``_draining`` is set first so new submissions are
        rejected immediately, then the worker is given a bounded window to
        finish the backlog. Cancelling the worker first would abandon requests
        that a client is still waiting on and that we already accepted
        responsibility for.
        """
        if not self._running:
            return
        self._draining = True
        logger.info("batcher draining", extra={"queue_depth": self._queue.qsize()})

        # Queue.join() returns once task_done() has been called for every item
        # ever queued. That is an event-driven wait: shutdown completes the
        # instant the backlog clears, instead of polling on a sleep loop that
        # both burns cycles and adds up to a tick of avoidable shutdown latency.
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._queue.join(), timeout=drain_timeout_s)

        remaining = self._queue.qsize()
        if remaining:
            logger.warning(
                "drain timed out with items still queued", extra={"abandoned": remaining}
            )

        self._running = False
        if self._worker is not None:
            self._worker.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker
            self._worker = None

        # Anything still queued will never be served; fail it explicitly rather
        # than leaving a client's future pending forever.
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:  # pragma: no cover - race guard
                break
            if not item.future.done():
                item.future.set_exception(
                    BatcherClosedError("service shut down before this request was served")
                )
            # Keep the queue's accounting balanced even on the abandon path.
            self._queue.task_done()
        logger.info("batcher stopped", extra={"batches": self._batches_processed})

    @property
    def running(self) -> bool:
        return self._running and not self._draining

    @property
    def model(self) -> ModelBackend:
        """The backend being served. Exposed so handlers can stamp responses
        with model identity without reaching into private state."""
        return self._model

    # -- submission ---------------------------------------------------------
    async def predict(self, text: str, *, timeout_s: float | None = None) -> Prediction:
        """Enqueue one input and await its prediction."""
        if not self._running or self._draining:
            raise BatcherClosedError("service is shutting down")

        loop = asyncio.get_running_loop()
        item = _BatchItem(text=text, future=loop.create_future())

        try:
            self._queue.put_nowait(item)
        except asyncio.QueueFull:
            self._items_rejected += 1
            if self._metrics is not None:
                self._metrics.record_rejection("queue_full")
            raise QueueFullError(
                f"inference queue is full ({self._max_queue_size} items); retry shortly"
            ) from None

        if self._metrics is not None:
            self._metrics.set_queue_depth(self._queue.qsize())

        budget = timeout_s if timeout_s is not None else self._request_timeout_s
        try:
            return await asyncio.wait_for(item.future, timeout=budget)
        except TimeoutError:
            self._items_timed_out += 1
            if self._metrics is not None:
                self._metrics.record_rejection("timeout")
            # The item may still be in the queue. Marking the future cancelled
            # lets the worker skip it instead of wasting a batch slot on a
            # caller that has already given up.
            item.future.cancel()
            raise InferenceTimeoutError(
                f"inference did not complete within {budget:.2f}s"
            ) from None

    async def predict_many(
        self, texts: Sequence[str], *, timeout_s: float | None = None
    ) -> list[Prediction]:
        """Submit several inputs concurrently.

        Each input is enqueued separately rather than forced into one batch, so
        a large client request cannot monopolise a batch slot and starve small
        ones. The scheduler decides the grouping.
        """
        tasks = [asyncio.create_task(self.predict(text, timeout_s=timeout_s)) for text in texts]
        try:
            return await asyncio.gather(*tasks)
        except BaseException:
            # One failure must not leave siblings running and unobserved.
            for task in tasks:
                task.cancel()
            raise

    # -- worker -------------------------------------------------------------
    async def _run(self) -> None:
        while True:
            try:
                batch = await self._collect_batch()
                if batch:
                    await self._process_batch(batch)
            except asyncio.CancelledError:
                raise
            except Exception:
                # If this task exits, every future queued afterwards hangs until
                # its timeout. Log and continue; a single bad batch is contained
                # in _process_batch, so reaching here means something unexpected.
                logger.exception("batcher worker loop error")
                await asyncio.sleep(0.01)

    async def _collect_batch(self) -> list[_BatchItem]:
        """Block for the first item, then window briefly for more.

        Two rules, and getting either wrong breaks batching in a way that only
        shows up under load:

        **1. Drain what is already queued, without waiting.** If a backlog
        exists there is nothing to wait *for* — the items are here now. Waiting
        would add latency and gain nothing.

        **2. The window is measured from when this batch started forming, not
        from when the head item was enqueued.** This is the subtle one. Under
        load an item may sit in the queue for far longer than ``max_wait_ms``
        before the worker reaches it. Deriving the deadline from its enqueue
        time makes that deadline already expired on arrival, so every batch
        closes with exactly one item — batching silently collapses precisely
        when it matters most, and the system gets *slower* the busier it is.

        Note the deadline is computed once, not refreshed per arrival. A
        refreshing deadline would let a steady arrival stream extend the window
        indefinitely and starve the item at the head. Fixing it here bounds
        head-of-line latency at ``max_wait_ms`` while still filling batches.
        """
        first = await self._queue.get()
        batch = [first]

        if self._max_batch_size == 1 or self._max_wait_s <= 0:
            return batch

        deadline = time.perf_counter() + self._max_wait_s

        while len(batch) < self._max_batch_size:
            # Rule 1: take anything already waiting, immediately.
            try:
                batch.append(self._queue.get_nowait())
                continue
            except asyncio.QueueEmpty:
                pass

            # Rule 2: nothing queued -- wait for a straggler, but only until the
            # fixed deadline.
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except TimeoutError:
                break
            batch.append(item)

        return batch

    async def _process_batch(self, batch: list[_BatchItem]) -> None:
        # Every item in the batch came from queue.get(), so every item owes a
        # task_done() -- including the dead ones dropped below. Missing even one
        # leaves Queue.join() waiting forever and turns graceful shutdown into a
        # guaranteed drain-timeout.
        try:
            await self._dispatch(batch)
        finally:
            for _ in batch:
                self._queue.task_done()

    async def _dispatch(self, batch: list[_BatchItem]) -> None:
        # Callers that already timed out or disconnected are dropped here rather
        # than consuming inference capacity.
        live = [item for item in batch if not item.future.done()]
        if self._metrics is not None:
            self._metrics.set_queue_depth(self._queue.qsize())
        if not live:
            return

        texts = [item.text for item in live]
        queue_times = [item.queue_time_seconds for item in live]
        started = time.perf_counter()

        try:
            # Off the event loop: see the module docstring.
            predictions = await asyncio.to_thread(self._model.predict_batch, texts)
        except Exception as exc:
            logger.exception("inference failed", extra={"batch_size": len(live)})
            for item in live:
                if not item.future.done():
                    item.future.set_exception(exc)
            if self._metrics is not None:
                self._metrics.record_inference_error(type(exc).__name__)
            return

        duration = time.perf_counter() - started

        # A backend returning the wrong number of results would silently pair
        # callers with other callers' predictions. Fail loudly instead.
        if len(predictions) != len(live):
            error = RuntimeError(
                f"backend contract violation: {len(live)} inputs produced "
                f"{len(predictions)} predictions"
            )
            logger.error(
                "backend returned mismatched batch length",
                extra={"inputs": len(live), "outputs": len(predictions)},
            )
            for item in live:
                if not item.future.done():
                    item.future.set_exception(error)
            return

        for item, prediction in zip(live, predictions, strict=True):
            if not item.future.done():
                item.future.set_result(prediction)

        self._batches_processed += 1
        self._items_processed += len(live)

        if self._metrics is not None:
            self._metrics.observe_batch(
                size=len(live), duration_seconds=duration, queue_times=queue_times
            )

        logger.debug(
            "batch processed",
            extra={
                "batch_size": len(live),
                "duration_ms": round(duration * 1000, 3),
                "queue_depth": self._queue.qsize(),
            },
        )

    # -- introspection ------------------------------------------------------
    def stats(self) -> BatcherStats:
        return BatcherStats(
            queue_depth=self._queue.qsize(),
            queue_capacity=self._max_queue_size,
            batches_processed=self._batches_processed,
            items_processed=self._items_processed,
            items_rejected=self._items_rejected,
            items_timed_out=self._items_timed_out,
            running=self.running,
        )
