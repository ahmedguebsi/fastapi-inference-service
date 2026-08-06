## What changed

<!-- One or two sentences. -->

## Why

<!-- The problem being solved, not the diff restated. -->

## How it was verified

<!-- Not "tests pass" -- what did you actually run, against what?
     e.g. "make check green; ran the container and confirmed graceful shutdown
     exits 0; benchmark still shows >20x batching speedup" -->

---

## Checklist

- [ ] `make check` passes locally
- [ ] New behaviour has tests that would fail without the change
- [ ] No new runtime dependency (each one ships in the image; open an issue first)
- [ ] `make docker-size` still within the 200 MB budget

**If this touches the serving layer (batcher, lifecycle, middleware):**

- [ ] Backpressure still rejects immediately rather than queueing without bound
- [ ] Requests already accepted are still drained on shutdown
- [ ] Readiness still flips to false *before* teardown begins
- [ ] `make bench` shows batching still engages under load (mean batch size > 1)

**If this touches metrics:**

- [ ] No unbounded label values — `endpoint` uses the route template, never the
      raw path, and `status` is a class not an exact code
- [ ] Zero-valued series are still emitted so alerts resolve rather than go silent
- [ ] `docs/metrics.md` updated

**If this touches the model or its artifact:**

- [ ] `python scripts/train_model.py` still reproduces the artifact byte-for-byte
- [ ] Artifact validation still rejects malformed input at load time
- [ ] `/v1/meta` reports the new `artifact_sha256`

**If this changes shutdown timing:**

- [ ] `prestop_delay_s + drain_timeout_s` still fits inside the documented
      orchestrator grace periods, and `docs/deployment.md` updated if not

## Risk

<!-- What could this break in production? How would you notice? How would you
     roll it back? "None" is fine if it is true. -->
