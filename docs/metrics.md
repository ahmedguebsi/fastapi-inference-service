# Metrics reference

Exposed at `/metrics` in Prometheus text format. Rendered from a per-app
`CollectorRegistry`, so nothing from the process-global default registry leaks in
and tests can build isolated instances.

## Cardinality rules

These are load-bearing, not stylistic. Unbounded label values are the most common
way to take down a Prometheus server.

| Label | Values | Why bounded |
| --- | --- | --- |
| `endpoint` | Route **template** (`/v1/predict`) | Never the raw path. A path parameter as a label value creates one series per distinct value. |
| `status` | `2xx`, `4xx`, `5xx` | Class, not exact code. Alerts are written against classes anyway. |
| `label` | Model's label set | Fixed by the artifact. |
| `reason` | `queue_full`, `timeout` | Closed enum. |
| `error_type` | Exception class name | Bounded by the code, not by input. |

Never labelled: request id, user text, API key, client IP.

---

## HTTP

### `inference_http_requests_total`
`counter` — labels `method`, `endpoint`, `status`.

```promql
# Error rate
sum(rate(inference_http_requests_total{status="5xx"}[5m]))
  / sum(rate(inference_http_requests_total[5m]))
```

### `inference_http_request_duration_seconds`
`histogram` — labels `method`, `endpoint`. End-to-end, including queue wait and
inference.

```promql
histogram_quantile(0.99,
  sum by (le, endpoint) (rate(inference_http_request_duration_seconds_bucket[5m])))
```

Buckets come from `observability.latency_buckets`, shaped for a ~100ms p99
target. Histogram buckets are cumulative and each is a time series — too few and
quantiles are meaningless, too many and cardinality suffers. Re-shape them if
your latency profile differs.

### `inference_http_requests_in_flight`
`gauge` — concurrent requests being served.

---

## Inference

### `inference_batch_size`
`histogram` — items per forward pass. **The single most informative metric in the
service.**

```promql
rate(inference_batch_size_sum[5m]) / rate(inference_batch_size_count[5m])
```

| Mean | Queue depth | Interpretation |
| --- | --- | --- |
| ≈ 1 | ~0 | Low load. Correct and healthy. |
| ≈ 1 | **high** | **Batching is not engaging** — bug or misconfiguration |
| mid | any | Healthy |
| ≈ max | high | Saturated. Scale out. |

The second row is the signature of the batching-collapse bug documented in
[architecture.md](architecture.md), where the batch window was measured from the
wrong instant and throughput *degraded* under load.

### `inference_batch_duration_seconds`
`histogram` — time inside `predict_batch`. Pure model time, excluding queueing.

Compare against queue wait to decide what to fix:

```promql
# queue wait >> inference time -> capacity problem, add replicas
histogram_quantile(0.99, rate(inference_queue_wait_seconds_bucket[5m]))
# inference time >> queue wait -> model problem, replicas will not help
histogram_quantile(0.99, rate(inference_batch_duration_seconds_bucket[5m]))
```

### `inference_queue_wait_seconds`
`histogram` — time an item spent queued before dispatch. Buckets are dense
sub-millisecond; a spike into the high buckets means saturation.

### `inference_queue_depth`
`gauge` — items waiting. The best autoscaling signal for this service (far better
than CPU, which a batched workload makes meaningless).

### `inference_batches_total`
`counter` — batches dispatched.

### `inference_predictions_total`
`counter` — label `label`. Predictions by predicted class.

```promql
# Prediction distribution -- a sudden shift can indicate model or data drift
sum by (label) (rate(inference_predictions_total[1h]))
  / ignoring(label) group_left sum(rate(inference_predictions_total[1h]))
```

### `inference_rejections_total`
`counter` — label `reason`.

- `queue_full` — backpressure. The design working; a **scaling** signal.
- `timeout` — the request budget expired. Either the model slowed or the queue
  is too deep relative to `request_timeout_s`.

Distinguishing these matters: one means add capacity, the other means something
got slower.

### `inference_inference_errors_total`
`counter` — label `error_type`. Exceptions raised inside the backend. **Always a
real defect** — this should be zero.

---

## Lifecycle

| Metric | Type | Meaning |
| --- | --- | --- |
| `inference_ready` | gauge | 1 when serving, 0 while starting or draining |
| `inference_start_time_seconds` | gauge | Unix timestamp of readiness |
| `inference_model_load_duration_seconds` | gauge | Artifact load time |
| `inference_model_info` | info | `name`, `version`, `backend`, `task`, `artifact_sha256` |
| `inference_build_info` | info | `version`, `service`, `environment` |

Info metrics are constant `1` with descriptive labels — the standard pattern for
attaching metadata without creating a series per changing value.

`inference_model_info` changing is how you detect an unannounced model rollout:

```promql
changes(inference_model_info[1h]) > 0
```

---

## PromQL you will actually use

```promql
# Throughput
sum(rate(inference_http_requests_total{endpoint=~"/v1/predict.*"}[5m]))

# p50 / p95 / p99 latency
histogram_quantile(0.50, sum by (le) (rate(inference_http_request_duration_seconds_bucket[5m])))
histogram_quantile(0.95, sum by (le) (rate(inference_http_request_duration_seconds_bucket[5m])))
histogram_quantile(0.99, sum by (le) (rate(inference_http_request_duration_seconds_bucket[5m])))

# Error rate excluding backpressure -- genuine failures only
sum(rate(inference_http_requests_total{status="5xx"}[5m]))
  - sum(rate(inference_rejections_total{reason="queue_full"}[5m]))

# Batching effectiveness
rate(inference_batch_size_sum[5m]) / rate(inference_batch_size_count[5m])

# Where the time goes
histogram_quantile(0.99, rate(inference_queue_wait_seconds_bucket[5m]))
histogram_quantile(0.99, rate(inference_batch_duration_seconds_bucket[5m]))

# Replicas currently unable to serve
count(inference_ready == 0)

# Capacity headroom
max(inference_queue_depth) / max(inference_batching_max_queue_size or vector(512))
```

---

## Alerting

Rules ship in [examples/alert_rules.yml](../examples/alert_rules.yml). The design
principles:

**Alert on symptoms, not causes.** Users feel latency and errors, not queue
depth. Queue depth is a *diagnostic*, useful as a leading indicator but not the
page.

**Separate broken from saturated.** `InferenceBackendErrors` means a defect and
should page whoever owns the model. `InferenceBackpressure` means insufficient
capacity and should trigger scaling. Conflating them sends the wrong person.

**Backpressure is not a bug.** 503 `queue_full` is the service protecting itself.
It is worth alerting on — sustained shedding means users are being turned away —
but at `warning`, as a scaling signal.

**Watch for silent degradation.** `InferenceBatchingIneffective` fires on mean
batch size ≈ 1 *while the queue is deep*. Nothing errors, nothing times out, the
service just quietly runs at a fraction of its capacity. That is the hardest
class of failure to notice and worth a dedicated rule.

---

## Multi-process note

If you run `workers > 1`, each process has its own registry and Prometheus
scrapes whichever one answers — so the numbers are per-worker and misleading.
`prometheus_client` supports multiprocess mode via `PROMETHEUS_MULTIPROC_DIR`,
but the simpler and better answer for model serving is to run one worker per
container and scale with replicas (see [performance.md](performance.md)). The
service logs a warning when `workers > 1`.
