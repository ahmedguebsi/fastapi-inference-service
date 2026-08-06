# API reference

Base URL: `http://localhost:8000`. Interactive docs at `/docs`, machine-readable
schema at `/openapi.json` (both disableable via `api.docs_enabled`).

## Conventions

**Versioning.** Inference endpoints live under `/v1/`. Operational endpoints
(`/health`, `/ready`, `/metrics`) are unversioned — they are contracts with your
orchestrator and monitoring, not with API clients, and they must not move when
the API version bumps.

**Correlation.** Every response carries `X-Request-ID`. Send your own and it is
honoured (truncated to 64 chars), so a trace spans an entire call chain. It
appears in every log line the request produced and in every error body.

**Timing.** `X-Process-Time-Ms` reports server-side handling time, so a client
can separate its own network latency from ours.

---

## `POST /v1/predict`

Classify a single text.

**Request**

```json
{ "text": "this service is excellent and works perfectly" }
```

| Field | Type | Constraints |
| --- | --- | --- |
| `text` | string | 1–`api.max_text_length` chars, not whitespace-only |

Unknown fields are rejected (`extra="forbid"`), so a typo like `"txt"` produces a
clear 422 rather than a confusing "field required" for a field you believe you
sent.

**Response `200`**

```json
{
  "result": {
    "label": "positive",
    "score": 0.937075,
    "scores": { "negative": 0.062925, "positive": 0.937075 }
  },
  "model_name": "sentiment",
  "model_version": "1.0.0",
  "request_id": "28d5ca28834a4ab89d4fc143f8c4d96e",
  "latency_ms": 10.962
}
```

`score` is the score of `label`, which is always the argmax of `scores`. For this
binary model the two scores sum to 1.

`model_name`/`model_version` are on every response deliberately: when a client
reports a bad prediction, the first question is *which model produced it*, and
you should not have to correlate timestamps against deploy logs to find out.

---

## `POST /v1/predict/batch`

Classify several texts in one call.

```json
{ "texts": ["fantastic quality", "broke within a week"] }
```

**Response `200`**

```json
{
  "results": [
    { "label": "positive", "score": 0.876, "scores": { "negative": 0.124, "positive": 0.876 } },
    { "label": "negative", "score": 0.816, "scores": { "negative": 0.816, "positive": 0.184 } }
  ],
  "model_name": "sentiment",
  "model_version": "1.0.0",
  "request_id": "481d899178e142218ba7c54f095904a1",
  "latency_ms": 24.509,
  "count": 2
}
```

Results are **positionally aligned** with `texts`.

> This endpoint is a client-side convenience, **not** the batching mechanism. The
> server submits each text to the scheduler independently, so a 64-item request
> cannot monopolise an inference batch and starve concurrent single-item callers.
> Server-side batching happens across *all* in-flight requests regardless of
> which endpoint they arrived on.

---

## `GET /health` · `/healthz` · `/live`

Liveness. **200 whenever the process is running and its event loop is
responsive.** Deliberately independent of model state.

```json
{ "status": "ok", "service": "inference-service", "version": "1.0.0", "uptime_seconds": 142.7 }
```

Point your orchestrator's *liveness* probe here. Failing it restarts the
container — which is the right response to a wedged process and the *wrong*
response to a failed model load (you would get a crash loop).

---

## `GET /ready` · `/readyz` · `/startup`

Readiness. 200 only when the model is loaded, the batcher is running, and the
service is not draining.

```json
{
  "status": "ready",
  "checks": { "model_loaded": true, "batcher_running": true, "not_shutting_down": true },
  "detail": null
}
```

**`503`** with `status: "not_ready"` and a `detail` explaining which check
failed. Failing this removes the instance from the load balancer *without*
restarting it.

Point your *readiness* probe here. During shutdown this flips to 503 first, so
the LB drains the instance before it stops accepting work.

---

## `GET /v1/meta`

Service and model identity, plus live scheduler statistics.

```json
{
  "service": "inference-service",
  "version": "1.0.0",
  "environment": "production",
  "model": {
    "name": "sentiment",
    "version": "1.0.0",
    "backend": "linear",
    "labels": ["negative", "positive"],
    "artifact_sha256": "9694cd21...",
    "parameters": 447,
    "load_duration_seconds": 0.0121
  },
  "batching": {
    "enabled": true,
    "max_batch_size": 32,
    "max_wait_ms": 10.0,
    "max_queue_size": 512,
    "request_timeout_s": 10.0
  },
  "batcher_stats": {
    "queue_depth": 0,
    "queue_capacity": 512,
    "batches_processed": 1284,
    "items_processed": 39112,
    "items_rejected": 0,
    "items_timed_out": 0,
    "mean_batch_size": 30.46,
    "running": true
  }
}
```

The first question in any inference incident is *"which model is actually
loaded?"*. `artifact_sha256` answers it without shell access, and lets you prove
which weights served a disputed prediction months later.

`mean_batch_size` is the fastest health check on the batching layer — near 1 with
a non-zero `queue_depth` means batching is not engaging.

---

## `GET /metrics`

Prometheus exposition. See [metrics.md](metrics.md). Excluded from the OpenAPI
schema: it is an operational endpoint, not part of the client contract.

---

## Errors

**One envelope for every failure**, so clients parse a single shape:

```json
{
  "error": {
    "code": "queue_full",
    "message": "The service is at capacity. Retry shortly.",
    "request_id": "3f9c1e08a4b74d2f9c1e08a4b74d2f9c",
    "details": null
  }
}
```

Branch on `code`, never on `message` — messages get reworded, codes are stable.

| Status | `code` | Meaning | Client should |
| ---: | --- | --- | --- |
| 401 | `unauthorized` | Missing/invalid `X-API-Key` | Fix credentials |
| 404 | `not_found` | No such path | Fix the URL |
| 405 | `method_not_allowed` | Wrong verb | Fix the method |
| 413 | `payload_too_large` | Over a configured limit | Split the request |
| 422 | `validation_error` | Body failed validation | Fix the body; see `details.fields` |
| 500 | `inference_failed` | Backend raised | Retry once, then escalate with `request_id` |
| 500 | `internal_error` | Unexpected | Escalate with `request_id` |
| 503 | `model_not_ready` | Still starting | Retry with backoff |
| 503 | `queue_full` | Backpressure | **Retry after `Retry-After`**, ideally another replica |
| 503 | `shutting_down` | Draining for deploy | Retry — another replica is available |
| 504 | `inference_timeout` | Exceeded the budget | Retry with backoff |

Validation errors include field-level detail:

```json
{
  "error": {
    "code": "validation_error",
    "message": "The request body failed validation.",
    "details": {
      "fields": [
        { "field": "text", "reason": "Field required", "type": "missing" }
      ]
    }
  }
}
```

**Error bodies never echo your input.** Pydantic's raw errors embed the offending
value; reflecting user text back would be both a PII leak and a reflection
vector, so only the location and reason are returned.

**500 responses never include a stack trace or internal path.** Those go to the
logs, keyed by `request_id`.

### Retry guidance

- **503 / 504** are safe to retry — the request either never reached the model or
  did not complete. Use exponential backoff with jitter, and honour `Retry-After`.
- **429** is not currently emitted; rate limiting is expected at the gateway.
- **422 / 413** are permanent. Retrying identical input will fail identically.

---

## Authentication

Disabled by default. Enable by configuring keys:

```bash
INFERENCE_API_KEYS=key-one,key-two inference-service
```

Then send `X-API-Key: key-one`. Comparison uses `secrets.compare_digest` and
checks every configured key without early exit, so response timing does not leak
which key matched or how many exist.

Health and readiness endpoints are **never** authenticated — an orchestrator that
cannot probe them will kill the pod.

---

## Generating a client

```bash
curl -s localhost:8000/openapi.json > openapi.json

# Python
openapi-python-client generate --path openapi.json

# TypeScript
npx openapi-typescript openapi.json -o client.ts
```

The schema documents all failure responses, so generated clients handle 503 and
504 as expected outcomes rather than surprises. CI asserts those stay documented.
