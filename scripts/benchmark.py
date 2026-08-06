#!/usr/bin/env python3
"""Measure the effect of dynamic batching on throughput and latency.

Runs the batching engine directly, without HTTP, so the numbers isolate the
scheduler rather than measuring uvicorn and the network stack.

    python scripts/benchmark.py
    python scripts/benchmark.py --concurrency 128 --requests 4000

Read the output as a *shape*, not an absolute: the shipped model is fast enough
that per-call fixed cost is small, so the batching win here is modest. With a
real model whose forward pass costs 20-50ms, the same curve is dramatic — that
is exactly why the batching layer exists. Use `--model-delay-ms` to simulate one.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from inference_service.model.base import ModelBackend, ModelInfo, Prediction
from inference_service.model.linear import LinearTextClassifier
from inference_service.serving.batcher import DynamicBatcher

DEFAULT_ARTIFACT = REPO_ROOT / "models" / "sentiment-v1.json"

SAMPLE_TEXTS = [
    "this product is excellent and i would recommend it to anyone",
    "terrible quality, broke within a week, complete waste of money",
    "does the job well enough, no complaints so far",
    "the support team was helpful and resolved my issue quickly",
    "disappointing performance and the documentation is incomplete",
]


class DelayedBackend(ModelBackend):
    """Wraps a real backend with a fixed per-CALL delay.

    Per call, not per item — that is the point. It simulates the fixed cost
    (kernel launch, memory transfer, framework overhead) that batching amortises
    in a real model runtime.
    """

    def __init__(self, inner: ModelBackend, delay_ms: float) -> None:
        self._inner = inner
        self._delay_s = delay_ms / 1000.0

    @property
    def info(self) -> ModelInfo:
        return self._inner.info

    def load(self) -> None:
        self._inner.load()

    def predict_batch(self, inputs) -> list[Prediction]:
        if self._delay_s:
            time.sleep(self._delay_s)
        return self._inner.predict_batch(inputs)


async def run_scenario(
    model: ModelBackend,
    *,
    total_requests: int,
    concurrency: int,
    max_batch_size: int,
    max_wait_ms: float,
) -> dict[str, float]:
    batcher = DynamicBatcher(
        model,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
        max_queue_size=max(1024, total_requests),
        request_timeout_s=120.0,
    )
    await batcher.start()

    latencies: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def one(index: int) -> None:
        async with semaphore:
            started = time.perf_counter()
            await batcher.predict(SAMPLE_TEXTS[index % len(SAMPLE_TEXTS)])
            latencies.append((time.perf_counter() - started) * 1000.0)

    started = time.perf_counter()
    await asyncio.gather(*(one(i) for i in range(total_requests)))
    wall = time.perf_counter() - started

    stats = batcher.stats()
    await batcher.stop(drain_timeout_s=5.0)

    latencies.sort()
    return {
        "throughput_rps": total_requests / wall,
        "wall_seconds": wall,
        "p50_ms": latencies[len(latencies) // 2],
        "p95_ms": latencies[int(len(latencies) * 0.95)],
        "p99_ms": latencies[int(len(latencies) * 0.99)],
        "mean_ms": statistics.fmean(latencies),
        "mean_batch_size": stats.mean_batch_size,
        "batches": float(stats.batches_processed),
    }


def format_row(label: str, result: dict[str, float]) -> str:
    return (
        f"{label:<22} "
        f"{result['throughput_rps']:>9.0f} "
        f"{result['mean_batch_size']:>7.1f} "
        f"{result['p50_ms']:>9.2f} "
        f"{result['p95_ms']:>9.2f} "
        f"{result['p99_ms']:>9.2f}"
    )


async def main_async(args: argparse.Namespace) -> int:
    base: ModelBackend = LinearTextClassifier(args.artifact)
    base.load()
    model = DelayedBackend(base, args.model_delay_ms) if args.model_delay_ms else base
    if args.model_delay_ms:
        model.load()

    print(f"requests={args.requests}  concurrency={args.concurrency}", end="")
    if args.model_delay_ms:
        print(f"  simulated per-call model cost={args.model_delay_ms}ms")
    else:
        print("  (real model, no simulated cost)")
    print()
    print(f"{'scenario':<22} {'rps':>9} {'batch':>7} {'p50 ms':>9} {'p95 ms':>9} {'p99 ms':>9}")
    print("-" * 72)

    scenarios = [
        ("batching disabled", 1, 0.0),
        ("batch 8 / wait 5ms", 8, 5.0),
        ("batch 32 / wait 10ms", 32, 10.0),
        ("batch 64 / wait 25ms", 64, 25.0),
    ]

    baseline: dict[str, float] | None = None
    for label, max_batch_size, max_wait_ms in scenarios:
        result = await run_scenario(
            model,
            total_requests=args.requests,
            concurrency=args.concurrency,
            max_batch_size=max_batch_size,
            max_wait_ms=max_wait_ms,
        )
        if baseline is None:
            baseline = result
        print(format_row(label, result))

    if baseline is not None:
        best = await run_scenario(
            model,
            total_requests=args.requests,
            concurrency=args.concurrency,
            max_batch_size=32,
            max_wait_ms=10.0,
        )
        speedup = best["throughput_rps"] / baseline["throughput_rps"]
        print("-" * 72)
        print(f"batch 32/10ms vs no batching: {speedup:.2f}x throughput")
        if speedup < 1.2 and not args.model_delay_ms:
            print(
                "\nNote: the shipped model is fast enough that per-call fixed cost is\n"
                "small, so batching has little to amortise. Re-run with\n"
                "  --model-delay-ms 20\n"
                "to see the curve a realistic model runtime produces."
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument(
        "--model-delay-ms",
        type=float,
        default=0.0,
        help="simulate a per-CALL model cost, as a real runtime would have",
    )
    args = parser.parse_args()

    if not args.artifact.is_file():
        print(f"artifact not found: {args.artifact}", file=sys.stderr)
        print("run: python scripts/train_model.py", file=sys.stderr)
        return 1

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
