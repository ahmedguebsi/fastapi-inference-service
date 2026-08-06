# syntax=docker/dockerfile:1.7
#
# Multi-stage build, targeting an image under 200 MB.
#
# ## Why Alpine
#
# `python:3.13-slim-bookworm` is ~190 MB uncompressed on its own. Adding the
# virtualenv puts the image at ~227 MB — over budget before a single line of
# application code. `python:3.13-alpine` is ~70 MB, landing the final image
# around 105 MB.
#
# Alpine uses musl instead of glibc, and that is normally a serious warning sign
# for Python: the scientific stack (numpy, torch, onnxruntime, scipy) publishes
# manylinux wheels but historically not musllinux, so those packages must be
# compiled from source — slow builds, and a genuine risk of subtle numerical
# differences.
#
# It is the right choice *here* specifically because this service has none of
# them. Every dependency — fastapi, uvicorn[standard] (uvloop, httptools),
# pydantic-core, prometheus-client — ships a musllinux wheel, verified in CI.
#
# **If you swap in an ONNX or PyTorch backend, revert this to slim-bookworm and
# raise the size budget.** That is a deliberate trade of image size for a real
# model, not a regression.
#
# ## Why multi-stage
#
# Stage 1 carries the full toolchain and produces a self-contained virtualenv.
# Stage 2 copies only that venv. The build toolchain, pip cache, tests and
# source tree never reach the final image, which both shrinks it and removes a
# large amount of attack surface — a compiler in a production image is a gift to
# anyone who achieves code execution.

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.13-alpine AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Metadata and source first, so Docker's layer cache survives edits elsewhere:
# changing a router does not re-resolve and re-download dependencies.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# --only-binary=:all: is a guard rail, not an optimisation. If a dependency ever
# stops publishing a musllinux wheel, this fails the build loudly instead of
# silently falling back to a source build that needs a compiler we deliberately
# did not install.
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel \
    && /opt/venv/bin/pip install --no-cache-dir --only-binary=:all: \
         fastapi "uvicorn[standard]" prometheus-client \
    && /opt/venv/bin/pip install --no-cache-dir --no-deps . \
    # Remove build tooling the runtime never uses. Keeps pip itself out of the
    # production image, which is a small but real supply-chain win.
    && /opt/venv/bin/pip uninstall -y pip setuptools wheel \
    && find /opt/venv -type d -name '__pycache__' -prune -exec rm -rf {} + \
    && find /opt/venv -type d -name 'tests' -prune -exec rm -rf {} + \
    && find /opt/venv -type f -name '*.pyc' -delete

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.13-alpine AS runtime

LABEL org.opencontainers.image.title="inference-service" \
      org.opencontainers.image.description="FastAPI text classification inference service with dynamic batching" \
      org.opencontainers.image.source="https://github.com/example/fastapi-inference-service" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.vendor="Ahmed Guebsi"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:${PATH}" \
    INFERENCE_HOST=0.0.0.0 \
    INFERENCE_PORT=8000 \
    INFERENCE_LOG_FORMAT=json \
    INFERENCE_ENVIRONMENT=production \
    INFERENCE_CONFIG=/etc/inference-service/service.toml \
    INFERENCE_MODEL_PATH=/opt/inference-service/models/sentiment-v1.json

# Unprivileged, no shell, no home. If the process is ever compromised, the
# account it runs as is worth nothing. (Alpine uses addgroup/adduser, not the
# shadow-utils groupadd/useradd found on Debian.)
RUN addgroup -S -g 65532 nonroot \
    && adduser -S -u 65532 -G nonroot -H -s /sbin/nologin nonroot

WORKDIR /opt/inference-service

COPY --from=builder --chown=root:root /opt/venv /opt/venv

# Model and config are root-owned and world-readable, but NOT writable by the
# runtime user. The service only reads them, so a compromised process must not
# be able to swap the model out from under itself.
COPY --chown=root:root models/ /opt/inference-service/models/
COPY --chown=root:root config/service.toml /etc/inference-service/service.toml

USER 65532:65532

EXPOSE 8000

# Probes READINESS, not liveness: a dependent service cares that the model is
# loaded and the batcher running, not merely that the process exists. urllib
# rather than curl, so the image needs no extra packages.
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=4).status==200 else 1)"]

# Exec form so the service is PID 1 and receives SIGTERM directly from
# `docker stop` / the kubelet. With the shell form, sh would be PID 1, would not
# forward the signal, and the container would be SIGKILLed after the grace
# period instead of draining its queue.
ENTRYPOINT ["inference-service"]
