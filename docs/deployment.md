# Deployment

## The setting that will bite you first

**Your orchestrator's kill grace period must exceed
`prestop_delay_s + drain_timeout_s`.**

Graceful shutdown takes that long by design. If the orchestrator kills the
process sooner, it is SIGKILLed mid-drain and you lose exactly the in-flight
requests the whole sequence exists to protect. The symptom is exit code **137**
during deploys, which is easy to misdiagnose as a crash.

Defaults are `prestop_delay_s = 5` + `drain_timeout_s = 30` = **35s**, which
exceeds Docker's default 10s and Kubernetes' default 30s.

| Platform | Setting | Set to |
| --- | --- | --- |
| `docker run` | `--stop-timeout` | 45 |
| Docker Compose | `stop_grace_period` | `45s` |
| Kubernetes | `terminationGracePeriodSeconds` | 45 |
| ECS | `stopTimeout` | 45 |

The service logs its shutdown budget at startup and warns when it exceeds 10s.

---

## Docker

```bash
docker build -t inference-service:1.0.0 .
docker run -d --name inference \
  -p 8000:8000 \
  --stop-timeout 45 \
  -e INFERENCE_ENVIRONMENT=production \
  -e INFERENCE_PRESTOP_DELAY_S=0 \
  inference-service:1.0.0
```

`INFERENCE_PRESTOP_DELAY_S=0` because a bare `docker run` has no load balancer to
deregister from. Behind one, set it above the health-check interval.

**Image:** 112 MB, Alpine-based, runs as uid 65532 with no login shell. See the
Dockerfile header for why Alpine — and why you should revert to
`slim-bookworm` if you swap in an ONNX or PyTorch backend.

**Verify a build:**

```bash
make docker-size                                    # enforces the 200MB budget
docker run --rm --entrypoint python IMAGE -c 'import os; print(os.getuid())'   # 65532
./scripts/smoke_test.sh http://localhost:8000
```

### Compose

```bash
docker compose up --build -d
open http://localhost:8000/docs
open http://localhost:9090        # Prometheus
open http://localhost:3000        # Grafana
```

---

## Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: inference-service
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0        # never drop below capacity during a deploy
  selector:
    matchLabels: { app: inference-service }
  template:
    metadata:
      labels: { app: inference-service }
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      # MUST exceed prestop_delay_s + drain_timeout_s, or the pod is SIGKILLed
      # mid-drain and in-flight requests are lost.
      terminationGracePeriodSeconds: 45

      securityContext:
        runAsNonRoot: true
        runAsUser: 65532
        runAsGroup: 65532
        fsGroup: 65532
        seccompProfile: { type: RuntimeDefault }

      containers:
        - name: api
          image: ghcr.io/example/inference-service:1.0.0
          ports:
            - { name: http, containerPort: 8000 }

          env:
            - { name: INFERENCE_ENVIRONMENT, value: production }
            - { name: INFERENCE_LOG_FORMAT,  value: json }
            # Above the readiness probe period (10s) so the endpoints
            # controller removes this pod before it stops accepting work.
            - { name: INFERENCE_PRESTOP_DELAY_S, value: "15" }
            - { name: INFERENCE_MAX_BATCH_SIZE,  value: "32" }
            - { name: INFERENCE_MAX_WAIT_MS,     value: "10" }
            - name: INFERENCE_API_KEYS
              valueFrom:
                secretKeyRef: { name: inference-api-keys, key: keys }

          # Liveness: restarts the container. Points at /health, which does NOT
          # depend on the model -- a failed model load must not cause a crash
          # loop.
          livenessProbe:
            httpGet: { path: /health, port: http }
            initialDelaySeconds: 10
            periodSeconds: 15
            timeoutSeconds: 3
            failureThreshold: 3

          # Readiness: removes the pod from the Service. Points at /ready, which
          # DOES depend on the model and on drain state.
          readinessProbe:
            httpGet: { path: /ready, port: http }
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 2

          # Startup: gives a slow model load a long budget without loosening
          # the liveness timeout for the pod's whole life.
          startupProbe:
            httpGet: { path: /ready, port: http }
            periodSeconds: 5
            failureThreshold: 24        # up to 2 minutes to load

          resources:
            requests: { cpu: "500m", memory: "256Mi" }
            limits:   { cpu: "2000m", memory: "512Mi" }

          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities: { drop: ["ALL"] }

          volumeMounts:
            - { name: tmp, mountPath: /tmp }

      volumes:
        - name: tmp
          emptyDir: { medium: Memory, sizeLimit: 64Mi }
```

### Why no `preStop` hook

A common pattern is `lifecycle.preStop: exec: ["sleep", "15"]` to let the
endpoints controller catch up. This service does it **in-process** instead, via
`prestop_delay_s`, and that is strictly better:

- The delay begins when the app knows it is shutting down, not when the kubelet
  gets around to running a hook.
- Readiness flips to false *immediately*, so deregistration starts at once and
  the sleep overlaps with it rather than following it.
- It works identically on ECS, Docker and bare metal, where `preStop` does not
  exist.

Add a `preStop` hook only if you also need to signal something outside the
process.

### Autoscaling

Scale on **queue depth**, not CPU. A batched inference workload can be at 60% CPU
and already shedding load, or at 95% and perfectly healthy — CPU is close to
meaningless as a proxy here.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: inference-service
spec:
  scaleTargetRef: { apiVersion: apps/v1, kind: Deployment, name: inference-service }
  minReplicas: 3
  maxReplicas: 30
  metrics:
    - type: Pods
      pods:
        metric: { name: inference_queue_depth }
        target: { type: AverageValue, averageValue: "20" }
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 30      # react quickly to load
    scaleDown:
      stabilizationWindowSeconds: 300     # shrink slowly, avoid flapping
```

Requires prometheus-adapter to surface `inference_queue_depth` as a pod metric.

---

## AWS ECS (Fargate)

```json
{
  "family": "inference-service",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "ACCOUNT.dkr.ecr.eu-west-1.amazonaws.com/inference-service:1.0.0",
      "portMappings": [{ "containerPort": 8000, "protocol": "tcp" }],
      "essential": true,
      "stopTimeout": 45,
      "user": "65532:65532",
      "readonlyRootFilesystem": true,
      "environment": [
        { "name": "INFERENCE_ENVIRONMENT", "value": "production" },
        { "name": "INFERENCE_LOG_FORMAT", "value": "json" },
        { "name": "INFERENCE_PRESTOP_DELAY_S", "value": "35" }
      ],
      "secrets": [
        { "name": "INFERENCE_API_KEYS", "valueFrom": "arn:aws:secretsmanager:...:inference/api-keys" }
      ],
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready',timeout=4).status==200 else 1)\""],
        "interval": 15,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 30
      },
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/inference-service",
          "awslogs-region": "eu-west-1",
          "awslogs-stream-prefix": "api"
        }
      }
    }
  ]
}
```

**ALB settings that matter:**

| Setting | Value | Why |
| --- | --- | --- |
| Health check path | `/ready` | Not `/health` — the LB wants readiness |
| Health check interval | 15s | `prestop_delay_s` must exceed this |
| Deregistration delay | 30s | Below `stopTimeout`, above the drain |
| Idle timeout | 60s | Must be **below** `keep_alive_timeout` (65s) |

That last row is the one that produces mystifying intermittent 502s: if the
server closes a connection the ALB still believes is open, the next request over
it fails. Keep-alive must outlive the LB's idle timeout, never the reverse.

---

## Configuration in production

Set via environment, not by baking a config file into the image — one artifact
must run in every environment.

```bash
INFERENCE_ENVIRONMENT=production
INFERENCE_LOG_FORMAT=json
INFERENCE_LOG_LEVEL=INFO
INFERENCE_DOCS_ENABLED=false        # unnecessary surface in production
INFERENCE_API_KEYS=<from secret manager>
INFERENCE_MAX_BATCH_SIZE=32
INFERENCE_MAX_WAIT_MS=10
INFERENCE_MAX_QUEUE_SIZE=512
INFERENCE_REQUEST_TIMEOUT_S=10
INFERENCE_PRESTOP_DELAY_S=15
```

Validate before rolling out — this exits non-zero on any problem and starts
nothing:

```bash
inference-service --check-config
```

`config/service.production.toml` is a worked example with the reasoning inline.

---

## Rollout checklist

1. `make check` and `make docker-size` green.
2. Deploy one canary replica. Confirm `/ready` returns 200 and `/v1/meta` shows
   the **expected `artifact_sha256`**.
3. Run `./scripts/smoke_test.sh https://canary-host` against it.
4. Watch for 10 minutes: `inference_batch_size` mean should be > 1 under load,
   `inference_inference_errors_total` should be 0, p99 within SLO.
5. Roll the rest with `maxUnavailable: 0`.
6. Confirm no exit-code-137 pods (`kubectl get pods` → check restart reasons). A
   137 during a rollout means the grace period is too short.

## Rollback

Model and code ship together, so rollback is a normal image rollback:

```bash
kubectl rollout undo deployment/inference-service
```

Verify with `/v1/meta` that `artifact_sha256` is back to the previous value. This
is exactly why the hash is exposed — "did the rollback actually revert the model?"
should not require reading deploy logs.
