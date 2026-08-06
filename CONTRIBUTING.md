# Contributing

## Setup

```bash
git clone https://github.com/example/fastapi-inference-service.git
cd fastapi-inference-service
make install
make check      # exactly what CI runs
```

Python 3.13+. `uv` is used when present, `pip` otherwise.

```bash
pre-commit install    # optional but recommended
```

## The gate

| Command | Enforces |
| --- | --- |
| `make fmt` | ruff format |
| `make lint` | ruff check, including bandit security rules |
| `make typecheck` | mypy `--strict` on `src/` |
| `make test` | pytest, 80% coverage floor |
| `make docker-size` | the 200 MB image budget |
| `make bench` | batching still engages under load |

Non-negotiables: **mypy --strict passes**, **no new runtime dependency without
discussion**, **the image stays under budget**.

Every runtime dependency ships in the production image and counts against that
budget. Open an issue before adding one.

## Adding a model backend

The most valuable contribution. Everything above `ModelBackend` — batching,
backpressure, timeouts, lifecycle, metrics — is already written and does not
change.

### 1. Implement the protocol

```python
# src/inference_service/model/onnx.py
class ONNXBackend(ModelBackend):
    BACKEND_NAME = "onnx"

    def __init__(self, artifact_path: str | Path) -> None: ...

    @property
    def info(self) -> ModelInfo: ...

    def load(self) -> None:
        """Fatal on failure. Validate strictly."""

    def predict_batch(self, inputs: Sequence[str]) -> list[Prediction]:
        """ONE vectorised call. Exactly one result per input, in order."""
```

Rules, each of which the serving layer depends on:

- **Synchronous.** The batcher runs it in a worker thread. A backend doing its
  own async I/O fights that design.
- **`predict_batch`, never `predict`.** Single prediction is a batch of one.
- **Exactly one result per input, positionally aligned.** The batcher resolves
  futures by position; a length mismatch would hand one caller another caller's
  prediction. It is checked at runtime and raises, but do not rely on that.
- **Thread-safe for reads after `load()`.** Multiple worker threads call
  `predict_batch` concurrently.
- **`load()` fails loudly.** Raise `ModelLoadError`. Startup aborts, which is
  correct — a pod that boots without a model and answers 503 forever is harder
  to diagnose than one that refuses to start.

### 2. Register it

```python
# src/inference_service/model/loader.py
BACKENDS = {
    LinearTextClassifier.BACKEND_NAME: LinearTextClassifier,
    ONNXBackend.BACKEND_NAME: ONNXBackend,
}
```

Add the name to `ModelConfig.validate`, and the dependency to an optional group:

```toml
[project.optional-dependencies]
onnx = ["onnxruntime>=1.20", "numpy>=2.0"]
```

### 3. Adjust the image

onnxruntime and numpy have **no musllinux wheels**. Revert the Dockerfile to
`python:3.13-slim-bookworm` and raise `SIZE_BUDGET_MB` in the Makefile and CI.
That is a deliberate trade of image size for a real model — document it in the
PR, do not do it silently.

### 4. Test it

Use `FakeBackend` in `tests/conftest.py` as the model. Backend-specific tests go
in `tests/test_model.py` and should cover: a valid artifact loads, a malformed
artifact raises `ModelLoadError`, batch results match individual results, and
inference is deterministic.

## Adding an endpoint

Handlers stay thin: validate, call the serving layer, map exceptions to the
error taxonomy. Scheduling logic belongs in `serving/`, not in a router.

- Define request/response models in `schemas.py` with `extra="forbid"`.
- Document failure responses in the `responses=` dict — CI asserts that
  `/v1/predict` keeps documenting 422, 503 and 504, because generated clients
  need to treat those as expected outcomes.
- New errors subclass `ServiceError` with a stable `code`. Clients branch on
  `code`; messages get reworded.
- Never let internal detail into a response body.

## Touching the serving layer

Highest-risk area in the repository. Read
[docs/architecture.md](docs/architecture.md) first.

Properties that must survive any change, each with a test that will catch you:

| Property | Test |
| --- | --- |
| Batching engages **more** under load, not less | `test_batching_does_not_collapse_under_backlog` |
| A backlog is drained without waiting out the window | `test_queued_items_are_drained_without_waiting` |
| Full queue rejects immediately, not after the timeout | `test_rejection_is_immediate_not_queued` |
| Accepted work is served during drain | `test_stop_drains_accepted_work` |
| Readiness flips false *before* teardown | `test_readiness_flips_before_teardown` |
| Results map to the correct caller | `test_results_map_to_the_correct_caller` |
| One bad batch does not kill the worker | `test_worker_survives_a_failing_batch` |

The batch window bug is worth internalising: measuring it from the head item's
*enqueue* time instead of from *batch formation* made batching collapse under
load and cost 27x throughput, while passing every unit test that existed at the
time. That is why CI now asserts end-to-end throughput, not just correctness.

## Metrics

- **Bounded labels only.** `endpoint` uses the route template, never the raw
  path. `status` is a class (`2xx`), never an exact code. A path parameter as a
  label value is the classic cardinality explosion that takes down Prometheus.
- **Never label** with request id, user text, API key or client IP.
- Emit zero-valued series so alerts resolve instead of going stale-silent.
- Update `docs/metrics.md`.

## Tests

- Unit tests use `FakeBackend`, which can be slow, can fail, and can violate the
  output-length contract. The real model is fast and always succeeds, which
  makes it useless for testing failure paths.
- Build apps through the `client` / `make_client` fixtures. They give each test
  an isolated `CollectorRegistry`; sharing one causes duplicate-timeseries
  errors.
- `LifespanManager` runs the real startup/shutdown path. A test that skips it is
  testing an app that can never serve a request.
- Keep `server.prestop_delay_s = 0` in test configs. It once made the suite take
  166s instead of 6s.
- `filterwarnings = ["error"]` is intentional. Fix the deprecation, do not mute
  it.

```bash
make test-fast                       # unit only
pytest tests/test_batcher.py -v
pytest -k "backpressure" -v
```

## Commits and PRs

Conventional Commits:

```
feat(model): add ONNX backend behind an optional dependency group
fix(batcher): measure the batch window from formation, not enqueue time
docs(performance): explain why batching lowers p50 under load
```

State what changed, why, and how you verified it — not "tests pass", but what
you actually ran and against what.

Before opening:

```bash
make check
make docker-size
make bench          # if you touched the serving layer
```

## Project layout

```
src/inference_service/
  main.py          App factory + lifespan. Startup/shutdown ORDERING lives here.
  config.py        TOML + env. Unknown keys fatal.
  dependencies.py  AppState and DI. No module-level globals.
  middleware.py    Request id, metrics, access log. Route templates only.
  errors.py        Error taxonomy; one envelope for every failure.
  schemas.py       Pydantic models; extra="forbid".
  metrics.py       Prometheus, bounded cardinality.
  logging_setup.py Structured logs + request-id contextvar.
  routers/         health · predict · meta. Thin.
  serving/
    batcher.py     THE serving layer. Queue, window, backpressure, drain.
  model/
    base.py        ModelBackend protocol. The seam.
    linear.py      Shipped backend.
    loader.py      Backend registry.
```

## Reporting security issues

Do not open a public issue. See [SECURITY.md](SECURITY.md).
