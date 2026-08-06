# Troubleshooting

Symptom-first. Each entry says how to confirm the cause before changing anything.

## Start here

```bash
curl -s localhost:8000/ready | jq          # why is it not serving?
curl -s localhost:8000/v1/meta | jq        # which model? how is the queue?
curl -s localhost:8000/metrics | grep -E 'inference_(ready|queue_depth|rejections)'
inference-service --check-config           # validate config, start nothing
```

`/v1/meta` answers the two questions that start most incidents: *which model is
loaded* (`artifact_sha256`) and *is the scheduler healthy* (`mean_batch_size`,
`queue_depth`).

Everything is correlated by `X-Request-ID`. Given one from a client:

```bash
kubectl logs -l app=inference-service --since=1h | grep <request-id>
```

---

## Startup

### The service exits immediately

Almost always a failed model load, which is **deliberately fatal**.

```
{"level":"ERROR","message":"model load failed; refusing to start","artifact_path":"..."}
```

```bash
ls -l models/sentiment-v1.json
python -c "import json; json.load(open('models/sentiment-v1.json'))"
python scripts/train_model.py     # regenerate
```

In a container, confirm the path the *container* sees:

```bash
docker run --rm --entrypoint sh IMAGE -c 'ls -l $INFERENCE_MODEL_PATH'
```

Booting without a model would be worse: the pod would sit in the rotation
answering 503 while looking alive.

### `unknown configuration key(s)`

Working as designed. A silently-ignored `max_batch_sze` means you believe you
tuned throughput and did not.

```bash
inference-service --check-config
diff config/service.toml config/service.production.toml
```

### `batching.max_wait_ms must be well below request_timeout_s`

The batch window would consume the entire request budget, so every request would
time out waiting to be batched. `max_wait_ms` is milliseconds;
`request_timeout_s` is seconds — mixing up the units is the usual cause.

### `server.prestop_delay_s must be below graceful_shutdown_timeout`

There would be no time left to drain the queue after waiting for the load
balancer.

---

## Requests

### Everything returns 503 `model_not_ready`

The model has not finished loading, or startup has not completed.

```bash
curl -s localhost:8000/ready | jq '.checks'
```

`model_loaded: false` after a minute means the load is hanging — check logs. If
you see it only right after a deploy, raise `startupProbe.failureThreshold`.

### 503 `queue_full`

**This is the design working, not a bug.** The queue is bounded and the service
is shedding load rather than accepting work it cannot finish.

```bash
curl -s localhost:8000/metrics | grep -E 'inference_(queue_depth|rejections_total)'
```

Fix in order of preference:

1. **Add replicas.** The real answer; you are out of capacity.
2. **Raise `max_batch_size`** if p99 latency has headroom — bigger batches mean
   more throughput per forward pass.
3. **Raise `max_queue_size`** only if requests are *not* also timing out. A
   deeper queue that just holds work until it expires is worse than a fast 503.

Do **not** remove the bound. Unbounded queueing under overload converts a fast
rejection into unbounded latency, memory growth, and an OOM kill that also
destroys the in-flight work.

### 504 `inference_timeout`

Requests are exceeding `request_timeout_s`. Find out where the time goes:

```promql
histogram_quantile(0.99, rate(inference_queue_wait_seconds_bucket[5m]))
histogram_quantile(0.99, rate(inference_batch_duration_seconds_bucket[5m]))
```

- Queue wait dominates → capacity problem. Add replicas, or lower
  `max_queue_size` so you reject early instead of timing out late.
- Inference time dominates → the model got slower. Check for a model change
  (`/v1/meta`), CPU throttling, or a noisy neighbour.

### 500 `inference_failed`

The backend raised. This is a genuine defect.

```bash
kubectl logs -l app=inference-service | grep -i 'inference backend error' | tail -5
curl -s localhost:8000/metrics | grep inference_inference_errors_total
```

The response deliberately contains no detail — quote the `request_id` and read
the logs.

### 422 on requests that look fine

Schemas use `extra="forbid"`, so an unknown field is rejected rather than
ignored.

```json
{"txt": "..."}   // rejected -- the field is "text"
{"text": "   "}  // rejected -- whitespace-only produces no tokens
```

`details.fields` names the offending field and reason. Note the response never
echoes your input, by design.

### 401 with a key that should work

```bash
curl -s localhost:8000/v1/meta | jq .service   # confirm you hit the right instance
```

Keys come from `INFERENCE_API_KEYS`, comma-separated. Common causes: whitespace
in the secret, the wrong header (`X-API-Key`, not `Authorization`), or one
replica in a rollout still carrying the old secret.

### Intermittent 502s from the load balancer, no matching server logs

Almost certainly **keep-alive mismatch**. If `server.keep_alive_timeout` is below
the LB's idle timeout, the server closes a connection the LB still believes is
open, and the next request over it fails before reaching the app — which is why
there is no server-side log.

```
keep_alive_timeout (65s)  >  ALB idle timeout (60s)   ✔
keep_alive_timeout (5s)   <  ALB idle timeout (60s)   ✘ intermittent 502s
```

---

## Performance

### Throughput is far below expectations

Check whether batching is engaging at all:

```promql
rate(inference_batch_size_sum[5m]) / rate(inference_batch_size_count[5m])
```

Mean ≈ 1 **while `inference_queue_depth` is high** means batching has collapsed —
the service is paying full fixed cost per item. This exact bug shipped once here
(window measured from enqueue time instead of batch formation) and cost 27x
throughput. See [architecture.md](architecture.md#2-the-batch-window-is-measured-from-batch-formation).

Also verify:

- `batching.enabled` is `true`.
- `max_batch_size > 1`.
- You are not running `workers > 1`, which splits arrivals across separate queues
  so each fills more slowly.

### Latency got worse after enabling batching

Under low load this is real and expected — you added up to `max_wait_ms`. Lower
it, or set it to 0 for an ultra-low-latency path.

Under **high** load it is almost never the window. Batching normally *reduces*
p50 substantially because it collapses queueing delay (measured: 1001ms → 34ms).
If latency rose under load, look at queue depth rather than the window.

### The first request after a deploy is slow

One-off initialisation. `model.warmup = true` should absorb it; confirm it is not
disabled. If the spike persists, the backend's `warmup()` is not exercising the
same path as a real request.

### High CPU but low throughput

Check `workers`. Each worker loads its own model copy and keeps its own queue, so
memory scales linearly and batches fill more slowly. Use one worker per container
and scale with replicas.

---

## Observability

### `/metrics` is empty or missing series

Counters only appear after their first observation — send a request first.
If the endpoint 404s, `observability.metrics_enabled` is false.

### Prometheus rejects the scrape

The content type must match the body format. This service emits Prometheus text
format and advertises `text/plain; version=1.0.0`. Advertising
`application/openmetrics-text` while emitting text format makes a conforming
scraper reject every scrape, because OpenMetrics requires a trailing `# EOF`.
(This shipped briefly here; there is now a test for it.)

```bash
curl -sI localhost:8000/metrics | grep -i content-type
```

### Logs are unparseable / doubled

`INFERENCE_LOG_FORMAT=json`. The service replaces uvicorn's handlers and disables
uvicorn's own access log, so each request produces exactly one structured line.
If you see two, something re-configured logging after startup.

### `KeyError: Attempt to overwrite 'name' in LogRecord`

A reserved `LogRecord` attribute was passed via `extra=`. `get_logger()` returns
an adapter that renames collisions to `ctx_*`, so this should be impossible — if
you see it, something is using `logging.getLogger` directly instead.

---

## Container

### Exit code 137 during deploys

**SIGKILL.** The orchestrator's grace period is shorter than
`prestop_delay_s + drain_timeout_s`, so the process was killed mid-drain and
in-flight requests were lost.

```bash
docker inspect <container> --format '{{.State.ExitCode}}'
```

Fix by raising the grace period, not by lowering the drain:

| Platform | Setting | Value |
| --- | --- | --- |
| `docker run` | `--stop-timeout` | 45 |
| Compose | `stop_grace_period` | `45s` |
| Kubernetes | `terminationGracePeriodSeconds` | 45 |

The service logs its shutdown budget at startup precisely so this is
discoverable before it bites.

### Container unhealthy on start

The healthcheck probes `/ready`, which needs the model loaded.
`--start-period=30s` covers a normal load; raise it for a slower model.

```bash
docker logs <container> | tail -30
```

### `read-only file system`

The compose file sets `read_only: true` and provides a `/tmp` tmpfs. That is
correct — the service writes nothing else. If you added something that does, give
it an explicit writable mount rather than disabling the protection.

---

## Development

### `pytest` hangs or takes minutes

`server.prestop_delay_s` is not 0 in the test config, so every test pays it on
teardown. The `base_config` fixture sets it to 0 for exactly this reason — it
once made the suite take 166s instead of 6s.

### `RuntimeError: Duplicate timeseries in CollectorRegistry`

Two apps sharing a registry. Use the `metrics` fixture, which builds a private
`CollectorRegistry` per test; that is why `create_app` takes a registry.

### Tests get 503 for everything

The lifespan did not run, so no model is loaded. Use the `client` fixture or wrap
the app in `LifespanManager` — httpx does not trigger lifespan events by itself.

### `StarletteDeprecationWarning` fails the suite

`filterwarnings = ["error"]` is intentional — it catches deprecations while they
are still cheap to fix. Update the call site rather than muting the warning.

---

## Escalating

Include:

```bash
inference-service --version
curl -s localhost:8000/v1/meta | jq
curl -s localhost:8000/ready | jq
curl -s localhost:8000/metrics | grep -E '^inference_(ready|queue_depth|rejections_total|inference_errors_total)'
kubectl logs -l app=inference-service --tail=100
```

Plus the `X-Request-ID` of a failing request. Review logs before sharing —
`log_request_body` and evidence fields can contain user text.
