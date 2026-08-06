# FastAPI Inference Service

A production-grade model serving API: dynamic request batching, bounded queueing
with backpressure, per-request timeouts, Prometheus instrumentation, structured
logging with request correlation, and a graceful shutdown that loses zero
in-flight requests.

[![CI](https://github.com/example/fastapi-inference-service/actions/workflows/ci.yml/badge.svg)](https://github.com/example/fastapi-inference-service/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Quick start

```bash
make install
make run          # http://localhost:8000/docs
```

```bash
curl -s localhost:8000/v1/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"this service is excellent and works perfectly"}' | jq
```

```json
{
  "result": {
    "label": "positive",
    "score": 0.907,
    "scores": { "negative": 0.093, "positive": 0.907 }
  },
  "model_name": "sentiment",
  "model_version": "1.0.0",
  "request_id": "3f9c1e08a4b74d2f9c1e08a4b74d2f9c",
  "latency_ms": 1.42
}
```

Container:

```bash
make docker
make demo         # service + Prometheus + Grafana
```

## What this actually demonstrates

The thing that separates an inference service from a CRUD API is **what happens
between the HTTP handler and the model**. This repository is about that layer.

### Dynamic batching

Model inference has a large fixed cost per call and a small marginal cost per
item. Serving one request per forward pass wastes most of that fixed cost.

Requests here do not call the model. They enqueue an item with a `Future` and
await it. A single worker drains the queue: take the first item, keep collecting
for up to `max_wait_ms` **or** until `max_batch_size` is reached, run one
`predict_batch`, resolve every future.

That window is the entire trade — latency for throughput, bounded and
configurable. Under low load, batches are size 1 and no measurable latency is
added. Under load, batches fill before the timeout and throughput climbs. Nothing
needs retuning between those two regimes.

The batching window is measured from when the **first** item arrived, not from
"now". Measuring from now lets a steady arrival stream extend the window forever
and starve the item at the head of the queue.

### Backpressure

The queue is bounded. Full queue means an immediate 503 with `Retry-After`.

An unbounded queue under sustained overload does not increase throughput. It
converts a fast, honest rejection into unbounded latency growth, memory
exhaustion, and eventually an OOM kill that takes the in-flight work with it. A
fast 503 lets the client retry another replica and lets the autoscaler see the
signal.

### Liveness ≠ readiness

`/health` answers *"is this process broken?"* — failing it **restarts** the
container. It never depends on the model: if the model fails to load, restarting
does not help, and you get a crash loop that turns a degradation into an outage.

`/ready` answers *"should traffic come here?"* — failing it **removes the pod
from the load balancer** but leaves it running. Model state, batcher state and
drain state belong here.

### Shutdown that loses nothing

On SIGTERM, in order:

1. Flip readiness to false **immediately** — the LB stops sending new work.
2. Wait out the LB's probe interval so in-flight routing decisions land.
3. Drain the queue: finish accepted work, reject new submissions.
4. Close the model.

Step 1 first is the whole trick. Tearing down before de-registering is the most
common cause of 502s during rolling deploys.

## About the model

The image budget is **under 200 MB**. A CPU-only PyTorch wheel is ~200 MB alone;
ONNX Runtime plus NumPy is ~80 MB before any weights. Either blows the budget.

So the shipped backend is a logistic regression over hashed n-grams, implemented
in pure Python with no third-party runtime dependency. The maths is real —
hashed bigram features, sublinear TF scaling, L2-normalised sparse dot product —
and [`scripts/train_model.py`](scripts/train_model.py) reproduces the artifact by
gradient descent, so the weights are not magic numbers.

**This is a deliberate scope decision, not a limitation being hidden.** Everything
that makes this a production service — batching, backpressure, timeouts,
lifecycle, drain, observability — is identical regardless of what sits behind
`ModelBackend`. Adding an ONNX backend is a new class plus an optional dependency
group, not a re-architecture. The sketch is in
[docs/architecture.md](docs/architecture.md#adding-a-real-model-backend).

The model is small and its accuracy is honestly reported by the training script.
Do not ship it to production and expect state-of-the-art sentiment analysis.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/predict` | Classify one text |
| `POST` | `/v1/predict/batch` | Classify several (scheduled independently) |
| `GET` | `/health`, `/healthz`, `/live` | Liveness |
| `GET` | `/ready`, `/readyz` | Readiness |
| `GET` | `/v1/meta` | Model identity, artifact hash, live scheduler stats |
| `GET` | `/metrics` | Prometheus exposition |
| `GET` | `/docs` | Swagger UI (disable in production) |

Every response carries `X-Request-ID` and `X-Process-Time-Ms`. Send your own
`X-Request-ID` and it is honoured, so a trace spans the whole call chain.

Full reference: [docs/api.md](docs/api.md).

### Errors

One envelope for every failure, so clients parse one shape:

```json
{
  "error": {
    "code": "queue_full",
    "message": "The service is at capacity. Retry shortly.",
    "request_id": "3f9c1e08a4b74d2f9c1e08a4b74d2f9c"
  }
}
```

| Status | Code | Meaning |
| ---: | --- | --- |
| 401 | `unauthorized` | Missing or invalid `X-API-Key` |
| 413 | `payload_too_large` | Text or batch exceeds the configured limit |
| 422 | `validation_error` | Body failed schema validation |
| 500 | `inference_failed` | Backend raised; details are in the logs, not the response |
| 503 | `model_not_ready` | Still starting |
| 503 | `queue_full` | Backpressure — retry |
| 503 | `shutting_down` | Draining |
| 504 | `inference_timeout` | Exceeded the time budget |

## Configuration

TOML plus `INFERENCE_*` environment overrides. Environment wins, so one image
runs everywhere. **Unknown keys are fatal** — a typo'd `max_batch_sze` must not
silently leave throughput untuned.

```bash
INFERENCE_MAX_BATCH_SIZE=64 \
INFERENCE_MAX_WAIT_MS=20 \
INFERENCE_API_KEYS=key-a,key-b \
  inference-service
```

Key knobs (full list in [config/service.toml](config/service.toml)):

| Setting | Default | Effect |
| --- | ---: | --- |
| `batching.max_batch_size` | 32 | Upper bound on items per forward pass |
| `batching.max_wait_ms` | 10 | The latency↔throughput dial |
| `batching.max_queue_size` | 512 | Backpressure threshold |
| `batching.request_timeout_s` | 10 | Per-request budget |
| `api.api_keys` | *(none)* | Empty disables auth |

## Metrics

| Metric | Type | Why you care |
| --- | --- | --- |
| `inference_http_request_duration_seconds` | histogram | End-to-end latency |
| `inference_batch_size` | histogram | Mean ≈1 → batching not engaging; ≈max → saturated |
| `inference_queue_wait_seconds` | histogram | Time queued before dispatch |
| `inference_queue_depth` | gauge | Backpressure proximity |
| `inference_batch_duration_seconds` | histogram | Pure model time |
| `inference_rejections_total` | counter | By reason: `queue_full`, `timeout` |
| `inference_ready` | gauge | 1 when serving |

Labels are strictly bounded — `endpoint` uses the **route template**, never the
raw path. A path parameter as a label value is the classic cardinality explosion
that takes down a Prometheus server.

Reference and PromQL: [docs/metrics.md](docs/metrics.md).

## Development

```bash
make help
make install      # venv + editable install + dev deps
make run          # reload server
make check        # everything CI runs
make test
make docker-size  # build and assert the 200MB budget
make bench        # local throughput benchmark
make train        # regenerate the model artifact
```

## Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Design decisions and their costs; ONNX backend sketch |
| [docs/api.md](docs/api.md) | Endpoint reference |
| [docs/metrics.md](docs/metrics.md) | Metric reference, PromQL, alerts |
| [docs/performance.md](docs/performance.md) | Tuning the batching window, benchmarks |
| [docs/deployment.md](docs/deployment.md) | Docker, Kubernetes, ECS, autoscaling |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Symptom-first runbook |
| [SECURITY.md](SECURITY.md) | Threat model |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to add a backend or endpoint |

## Known limitations

- **Single-node batching.** The queue is per-process. Multiple uvicorn workers
  each hold their own model copy and their own queue, which fragments batches —
  prefer horizontal replicas over `workers > 1`. The service warns if you do it.
- **No request priority.** FIFO. A latency-critical caller cannot jump ahead of
  a bulk one. Priority queueing is a natural extension.
- **No streaming.** Request/response only, which suits classification. Token
  streaming for generative models needs a different response path.
- **The shipped model is small.** By design, for the image budget. See above.

## License

MIT — see [LICENSE](LICENSE).
