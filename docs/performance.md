# Performance and tuning

## The one number that matters: `max_wait_ms`

Everything else is secondary. This is how long the scheduler waits, from the
moment a batch starts forming, for more items to join it — the maximum latency
batching can add to any request.

```
lower   ->  smaller batches, lower latency, lower peak throughput
higher  ->  fuller batches, higher throughput, higher tail latency
```

Crucially, this is **not** a fixed latency cost. When a backlog exists the
scheduler drains it with `get_nowait()` and waits zero. The window only applies
when the queue runs dry mid-batch, i.e. under light load — exactly when you have
latency budget to spare.

## Measured effect

`scripts/benchmark.py` drives the batcher directly, without HTTP, so the numbers
isolate the scheduler. `--model-delay-ms` simulates the per-**call** fixed cost a
real model runtime has (kernel launch, memory transfer, framework overhead).

```bash
python scripts/benchmark.py --requests 1200 --concurrency 64 --model-delay-ms 15
```

```
requests=1200  concurrency=64  simulated per-call model cost=15.0ms

scenario                     rps   batch    p50 ms    p95 ms    p99 ms
------------------------------------------------------------------------
batching disabled             64     1.0   1000.68   1005.28   1007.92
batch 8 / wait 5ms           498     8.0    128.14    129.49    129.75
batch 32 / wait 10ms        1791    31.6     34.12     34.78     47.56
batch 64 / wait 25ms        3079    63.2     18.51     20.44     44.78
```

Read that carefully, because it is counter-intuitive:

**Batching made latency 29x *better*, not worse.** p50 went from 1001ms to 34ms.

The naive mental model says batching trades latency for throughput. That is true
only at low load. Under load the dominant term is **queueing delay**, not the
batch window — and queueing delay is inversely proportional to throughput. A
service that processes 64 rps with 64 concurrent clients has each client waiting
behind a full second of queue. Raise throughput to 1791 rps and the queue
evaporates.

Batching is a latency optimisation under load. The window cost (≤10ms) is noise
next to the queueing it eliminates (~966ms).

### Without a simulated model cost

```bash
python scripts/benchmark.py --requests 2000 --concurrency 64
```

The shipped linear model runs in microseconds, so per-call fixed cost is tiny and
there is almost nothing for batching to amortise — speedup is near 1x. That is
the honest result and the benchmark says so in its output. It is also why
`--model-delay-ms` exists: it shows the curve any realistic model runtime
produces.

## Choosing values

Start from your model's single-item latency `L`:

| Model latency `L` | `max_wait_ms` | `max_batch_size` | Reasoning |
| --- | ---: | ---: | --- |
| < 1ms (linear, tiny) | 0–5 | 8–16 | Little fixed cost to amortise; keep it simple |
| 5–20ms (small transformer, CPU) | 10 | 32 | The shipped defaults |
| 20–100ms (BERT-base, CPU) | 25–50 | 32–64 | Fixed cost dominates; wait pays for itself |
| > 100ms (GPU, large model) | 50–100 | 64–256 | Batch hard; GPUs are throughput devices |

Rules of thumb:

- **`max_wait_ms` ≈ `L`.** Waiting much longer than one inference costs more
  than it saves.
- **`max_batch_size`:** raise until p99 latency starts rising faster than
  throughput. That inflection is your hardware's saturation point.
- **`max_queue_size` ≈ `peak_rps × request_timeout_s`.** Beyond that, items
  expire before reaching the model — you are storing work only to throw it away.
- **`request_timeout_s`** should be *below* your client's timeout, so the client
  gets a clean 504 with a request id rather than an opaque connection reset.

## Reading the metrics

The three that tell you what to change:

**`inference_batch_size`** — mean batch size.

```promql
rate(inference_batch_size_sum[5m]) / rate(inference_batch_size_count[5m])
```

| Observation | Meaning | Action |
| --- | --- | --- |
| ≈ 1, queue empty | Low load | Nothing. Correct. |
| ≈ 1, **queue deep** | **Batching is not engaging** | Bug or misconfiguration — see below |
| ≈ `max_batch_size` | Saturated | Add replicas, or raise the cap if latency allows |
| Mid-range | Healthy | Nothing |

Mean batch size near 1 *while the queue is deep* is the signature of the
batching-collapse bug documented in [architecture.md](architecture.md) — the
window measured from the wrong instant. `InferenceBatchingIneffective` in
`examples/alert_rules.yml` alerts on exactly this conjunction.

**`inference_queue_wait_seconds` vs `inference_batch_duration_seconds`** — where
the time actually goes.

```promql
histogram_quantile(0.99, rate(inference_queue_wait_seconds_bucket[5m]))
histogram_quantile(0.99, rate(inference_batch_duration_seconds_bucket[5m]))
```

- Queue wait ≫ inference time → **capacity problem**. Add replicas.
- Inference time ≫ queue wait → **model problem**. Optimise, quantise, or use
  better hardware. More replicas will not help per-request latency.

**`inference_rejections_total`** — what kind of overload.

- `reason="queue_full"` → backpressure. Working as designed; scale out.
- `reason="timeout"` → items expiring in the queue. Either the model slowed
  down, or `max_queue_size` is too large relative to `request_timeout_s`.

## Scaling

**Replicas, not workers.** Each uvicorn worker is a separate process with its own
model copy and its own queue. That means memory scales linearly *and* arrivals
are split across queues, so batches fill more slowly — you pay twice. The service
logs a warning when `workers > 1`.

Horizontal replicas behind a load balancer are strictly better for model serving:
same memory-per-model, no queue fragmentation, and independent failure domains.

**Autoscale on queue depth or rejections, not CPU.** CPU utilisation is a poor
proxy for an inference service — a batched workload can be at 60% CPU and already
rejecting requests, or at 95% and perfectly healthy.

```yaml
# HorizontalPodAutoscaler, external metric
metrics:
  - type: Pods
    pods:
      metric:
        name: inference_queue_depth
      target:
        type: AverageValue
        averageValue: "20"
```

## Benchmarking honestly

Things that will make your numbers lie:

- **Not warming up.** The first request pays one-off costs. `model.warmup` is on
  by default for this reason; a benchmark that skips it measures initialisation.
- **Measuring mean.** Report p50/p95/p99. Mean latency hides the tail that
  actually pages you.
- **Testing on a laptop.** Thermal throttling makes later runs slower than
  earlier ones, which reads as a regression.
- **Benchmarking through HTTP when you mean to benchmark the scheduler.**
  `scripts/benchmark.py` deliberately bypasses uvicorn. To measure the full
  stack, use a real load generator against a real deployment.
- **Ignoring the client.** At high rps a Python load generator becomes the
  bottleneck and you end up benchmarking your own test harness.

For end-to-end HTTP load testing:

```bash
# k6 -- includes think time and connection reuse, closer to real traffic
k6 run --vus 50 --duration 60s examples/load_test.js

# hey -- quick and dirty
hey -n 5000 -c 50 -m POST \
  -H 'Content-Type: application/json' \
  -d '{"text":"benchmark this"}' \
  http://localhost:8000/v1/predict
```
