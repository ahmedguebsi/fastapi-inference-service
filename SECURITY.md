# Security Policy

## Reporting a vulnerability

Do **not** open a public issue.

Email **ahmed.guebsi@enis.tn** with `[SECURITY]` in the subject, or use GitHub's
private vulnerability reporting on this repository.

Include the affected version, a description, reproduction steps, and the impact
you believe it has. Acknowledgement within 72 hours, assessment within 7 days,
fixes for confirmed high/critical issues targeted within 30 days. You will be
credited in the advisory unless you prefer otherwise.

## Threat model

This is an internet-adjacent HTTP service that executes a model over
attacker-supplied text. The realistic threats are resource exhaustion, input
abuse, and information disclosure — not remote code execution.

| Threat | Mitigation |
| --- | --- |
| Resource exhaustion via request flood | Bounded queue with immediate 503 + `Retry-After`. The service sheds load rather than accepting work it cannot finish. |
| Resource exhaustion via huge payloads | `max_text_length` and `max_texts_per_request` enforced at the schema **and** in the router, before anything reaches the queue. |
| Slow-loris / hung requests | Per-request timeout (`request_timeout_s`); uvicorn keep-alive timeout. |
| Unbounded memory growth | Queue capacity is fixed; container memory limits in compose and k8s manifests. |
| Credential leakage via timing | API keys compared with `secrets.compare_digest`, every candidate checked with no early exit, so timing reveals neither which key matched nor how many exist. |
| User text leaking into logs | `log_request_body` off by default; error responses never echo input. |
| Internal detail leaking to clients | 500 responses carry a `request_id` and nothing else. Stack traces and paths go to logs only. |
| Reflection via validation errors | Pydantic's raw errors embed the offending value; the handler strips it and returns only field location and reason. |
| Metrics cardinality attack | `endpoint` labels use the route template, never the raw path. Unmatched paths are truncated to 64 chars. |
| Compromised container | Runs as uid 65532, no login shell, `cap_drop: ALL`, `no-new-privileges`, read-only root filesystem, model and config root-owned and not writable by the runtime user. |
| Supply chain | Three runtime dependencies, all pinned by a lockfile in CI. Trivy scans filesystem and image; Gitleaks scans for secrets; Dependabot tracks pip, Actions and Docker. |

### Explicitly out of scope

- **Authentication and authorisation beyond a shared API key.** This service is
  designed to sit behind a gateway that does real authn/z. The API key is a
  defence-in-depth backstop, not an identity system.
- **Rate limiting per client.** Backpressure protects the *service*; it does not
  fairly apportion capacity between clients. Do that at the gateway.
- **TLS.** Terminate at the load balancer or service mesh.
- **Model-level attacks.** Adversarial inputs, prompt injection, model
  extraction and membership inference are real concerns for ML systems and are
  not addressed here.
- **Confidentiality of predictions in transit.** No response signing.

## Handling of user data

**What the service receives:** arbitrary text submitted for classification.

**What it does with it:**

- Holds it in memory for the duration of the request.
- Passes it to the model.
- Discards it. Nothing is persisted; there is no database and no disk write.

**What it never does:**

- Never logs request bodies unless `log_request_body` is explicitly enabled.
- Never echoes input in error responses.
- Never includes text in metric labels.
- Never transmits anything off-host. No telemetry, no phone-home.

**Where care is still needed:** enabling `log_request_body`, or setting
`logging.level = "DEBUG"`, will put user text in your log backend. Both require a
lawful basis and a retention policy.

## Deployment hardening

The shipped defaults are safe; these are the things to verify in your
environment.

```bash
# Container is unprivileged
docker run --rm --entrypoint python IMAGE -c 'import os; print(os.getuid())'   # 65532

# Runtime user has no login shell
docker run --rm --entrypoint python IMAGE \
  -c "import pwd,os; print(pwd.getpwuid(os.getuid()).pw_shell)"                # /sbin/nologin

# Interactive docs disabled in production
curl -s -o /dev/null -w '%{http_code}' https://host/docs                        # 404

# Auth is on
curl -s -o /dev/null -w '%{http_code}' -XPOST https://host/v1/predict \
  -H 'Content-Type: application/json' -d '{"text":"x"}'                         # 401
```

Checklist:

- [ ] `INFERENCE_DOCS_ENABLED=false`
- [ ] `INFERENCE_API_KEYS` sourced from a secret manager, never an image layer
- [ ] `/metrics` not exposed publicly — it reveals traffic volume, model
      identity and saturation state
- [ ] TLS terminated upstream
- [ ] Rate limiting at the gateway
- [ ] Container memory and CPU limits set, so a leak becomes a restart rather
      than a node-level OOM
- [ ] `readOnlyRootFilesystem: true`

## Supply chain

- Three runtime dependencies: `fastapi`, `uvicorn[standard]`, `prometheus-client`.
- Dev dependencies never ship in the image (multi-stage build).
- `pip`, `setuptools` and `wheel` are removed from the runtime venv.
- The builder installs with `--only-binary=:all:`, so a dependency that stops
  publishing a wheel fails the build loudly instead of silently invoking a
  compiler that is deliberately absent.
- Release images carry SLSA provenance attestation and an SBOM.
- The model artifact's SHA-256 is recorded at load and exposed on `/v1/meta`, so
  you can prove which weights served a given prediction.

Verify a published image:

```bash
gh attestation verify oci://ghcr.io/example/inference-service:1.0.0 --owner example
```

## Supported versions

| Version | Supported |
| --- | --- |
| 1.0.x | Yes |
| < 1.0 | No |

## Scope

**In scope:** authentication bypass, resource exhaustion not mitigated by the
documented limits, information disclosure in responses or logs, metrics
cardinality attacks, container escape, dependency vulnerabilities reachable from
request handling.

**Out of scope:** the items listed under *Explicitly out of scope* above;
findings requiring existing host access; vulnerabilities in Python itself (report
upstream); `/metrics` or `/docs` being reachable because an operator exposed them
against documented advice.
