# Examples

| File | What it is |
| --- | --- |
| [`requests.http`](requests.http) | Request collection for VS Code REST Client / JetBrains HTTP Client |
| [`prometheus.yml`](prometheus.yml) | Prometheus config for the compose stack |
| [`alert_rules.yml`](alert_rules.yml) | Production-shaped alerting rules |
| [`grafana-datasource.yml`](grafana-datasource.yml) | Grafana provisioning for the demo |

## Running the demo

```bash
docker compose up --build -d

open http://localhost:8000/docs   # Swagger UI
open http://localhost:9090        # Prometheus
open http://localhost:3000        # Grafana (anonymous)
```

Generate some load, then watch the batching engage:

```bash
for i in $(seq 1 200); do
  curl -s localhost:8000/v1/predict \
    -H 'Content-Type: application/json' \
    -d '{"text":"this is a load generating request"}' > /dev/null &
done; wait
```

In Prometheus:

```promql
# Mean batch size -- should rise above 1 under concurrent load
rate(inference_batch_size_sum[1m]) / rate(inference_batch_size_count[1m])

# Where the time actually goes
histogram_quantile(0.99, rate(inference_queue_wait_seconds_bucket[1m]))
histogram_quantile(0.99, rate(inference_batch_duration_seconds_bucket[1m]))
```

Tear down with `docker compose down -v`.

## Alert rule design

The rules in `alert_rules.yml` follow four principles worth reusing:

1. **Alert on symptoms, not causes.** Users feel latency and errors. Queue depth
   is a diagnostic and a leading indicator, not the page.
2. **Separate broken from saturated.** `InferenceBackendErrors` means a defect
   and should page whoever owns the model. `InferenceBackpressure` means
   insufficient capacity and should trigger scaling. Conflating them routes the
   incident to the wrong person.
3. **Backpressure is the design working.** 503 `queue_full` is the service
   protecting itself. Worth alerting at `warning` — sustained shedding means
   users are being turned away — but it is not a bug.
4. **Catch silent degradation.** `InferenceBatchingIneffective` fires when mean
   batch size sits near 1 *while the queue is deep*. Nothing errors, nothing
   times out; the service just quietly runs at a fraction of its capacity. That
   is the hardest failure class to notice, and it shipped here once — see
   [../docs/architecture.md](../docs/architecture.md).

Load them into a real Prometheus:

```bash
cp alert_rules.yml /etc/prometheus/
promtool check rules /etc/prometheus/alert_rules.yml
curl -X POST http://localhost:9090/-/reload
```

## Benchmarking

`scripts/benchmark.py` drives the batcher directly, bypassing HTTP, so the
numbers isolate the scheduler:

```bash
python scripts/benchmark.py --requests 1200 --concurrency 64 --model-delay-ms 15
```

`--model-delay-ms` simulates the per-**call** fixed cost a real model runtime
has. Without it the shipped linear model is fast enough that there is almost
nothing for batching to amortise — which the benchmark says out loud rather than
flattering itself. See [../docs/performance.md](../docs/performance.md).
