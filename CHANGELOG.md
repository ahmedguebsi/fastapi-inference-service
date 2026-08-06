# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Nothing yet.

## [1.0.0] - 2026-08-06

First production release.

### Added

**Serving layer**

- Dynamic request batching: concurrent requests are grouped into a single model
  call, bounded by `max_batch_size` and a `max_wait_ms` window.
- Bounded queue with immediate 503 + `Retry-After` on overflow, so overload
  produces fast rejection rather than unbounded latency and an eventual OOM.
- Per-request timeouts, with abandoned items filtered out before they consume
  inference capacity.
- Model inference dispatched to a worker thread, keeping the event loop
  responsive (real parallelism for backends that release the GIL).
- Backend contract enforcement: a batch returning the wrong number of results
  raises rather than silently pairing callers with other callers' predictions.
- Worker isolation — one failing batch cannot kill the scheduler.

**API**

- `POST /v1/predict`, `POST /v1/predict/batch`.
- `GET /health` · `/healthz` · `/live` — liveness, independent of model state.
- `GET /ready` · `/readyz` · `/startup` — readiness, gating the load balancer.
- `GET /v1/meta` — model identity including artifact SHA-256, plus live
  scheduler statistics.
- `GET /metrics` — Prometheus exposition.
- One error envelope for every failure, with stable machine-readable codes.
- Pydantic v2 schemas with `extra="forbid"`; limits enforced before work is
  queued.
- Optional API-key auth with constant-time comparison and no early exit.

**Lifecycle**

- Ordered startup: load model (fatal on failure) → warm up → start batcher →
  advertise ready.
- Ordered shutdown: flip readiness false → wait `prestop_delay_s` for load
  balancer deregistration → drain accepted work → close model.
- Startup log announces the shutdown budget and warns when it exceeds 10s, since
  an orchestrator grace period shorter than that causes a SIGKILL mid-drain.

**Observability**

- Prometheus metrics for HTTP, batching, queueing, rejections and lifecycle,
  with strictly bounded label cardinality (route templates, status classes).
- Structured JSON logging with `request_id` propagated via `contextvars`, so
  every line from a request correlates — including lines from inside the
  batching engine.
- Client-supplied `X-Request-ID` honoured for cross-service tracing.
- A logger adapter that renames `extra` keys colliding with `LogRecord`
  attributes, so a log call can never crash a request.

**Model**

- Pluggable `ModelBackend` protocol with batch-first inference.
- Linear text classifier over hashed n-grams: sublinear TF scaling, L2
  normalisation, sparse dot product, numerically stable sigmoid.
- Stable BLAKE2b feature hashing, not Python's per-process randomised `hash()`,
  so replicas of one deployment agree on predictions.
- Strict artifact validation; SHA-256 recorded and exposed.
- `scripts/train_model.py` reproduces the artifact byte-for-byte (CI asserts it).

**Packaging and operations**

- Multi-stage Alpine image, **112 MB**, running as uid 65532 with no login
  shell, read-only root filesystem and all capabilities dropped.
- `docker-compose.yml` with Prometheus and Grafana, and a `stop_grace_period`
  that actually accommodates the drain.
- CI: lint, typecheck, tests on Ubuntu and macOS, model reproducibility,
  OpenAPI contract, shellcheck, Trivy, Gitleaks, image size budget, non-root
  assertion, container smoke test, graceful-shutdown exit-code assertion, and a
  batching throughput gate.
- 178 tests; `mypy --strict`; ruff with bandit rules.

### Fixed during development

Recorded because the reasoning is the useful part.

- **Batching collapsed under load.** The batch window was measured from the head
  item's *enqueue* time. Once a backlog formed, that deadline was already
  expired on dequeue, so every batch closed with one item — throughput
  *degraded* as load rose. Now measured from batch formation, with queued items
  drained via `get_nowait()` before any waiting. Benchmark: **66 → 1791 rps**
  (27x) and p50 latency 1001ms → 34ms. Caught by the benchmark, not by the unit
  tests, which is why CI now gates on throughput.
- **Log call crashed startup.** `extra={"name": ...}` collides with
  `LogRecord.name` and raises `KeyError` from inside `logging`. Fixed at the
  call site and hardened with an adapter so it cannot recur.
- **Metrics content type did not match the body.** Served Prometheus text format
  while advertising `application/openmetrics-text`; a conforming scraper rejects
  that, because OpenMetrics requires a trailing `# EOF`. Now paired correctly,
  with a test.
- **Image over budget.** Debian slim base put the image at 227 MB. Moved to
  Alpine (112 MB) after verifying every dependency publishes a musllinux wheel.
- **Test suite took 166s.** A hardcoded 5s pre-stop delay ran on every test
  teardown. Made configurable (`prestop_delay_s`) and set to 0 in tests: 6s.

### Design decisions

- **Batch is the primitive.** `predict_batch` is the only inference method, so a
  vectorising backend needs no change to the serving layer.
- **Liveness never depends on the model.** A failed model load must remove a pod
  from service, not restart it forever.
- **Readiness flips false before teardown.** Tearing down before de-registering
  is the most common cause of 502s during rolling deploys.
- **`prometheus_client` is a dependency here**, unlike the hardening agent which
  hand-rolls exposition. Different privilege context, different answer — see
  `docs/architecture.md`.
- **Small model, by design.** The 200 MB image budget excludes PyTorch and ONNX
  Runtime. The serving infrastructure is the deliverable and is unchanged by
  what sits behind `ModelBackend`.

### Known limitations

- Single-node batching; the queue is per-process. Prefer replicas over
  `workers > 1`, which fragments batches and duplicates the model in memory.
- FIFO only — no request priority.
- No streaming responses.
- `max_wait_ms` is static; there is no adaptive controller.
- The shipped model is small and its accuracy is reported honestly by the
  training script.

[Unreleased]: https://github.com/example/fastapi-inference-service/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/example/fastapi-inference-service/releases/tag/v1.0.0
