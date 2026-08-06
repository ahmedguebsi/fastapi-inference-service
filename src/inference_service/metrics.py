"""Prometheus instrumentation.

**Why take the ``prometheus_client`` dependency here when the hardening agent
rendered exposition by hand?** The calculus is different, and it is worth being
explicit about that rather than applying a rule mechanically.

The hardening agent runs *as root, on every host in a fleet*. Every dependency
there is a fleet-wide patching obligation and a supply-chain path into the most
privileged process on the box, so the bar for taking one is extremely high.

This service runs unprivileged, in a container, already depending on FastAPI,
Uvicorn and Pydantic. One more well-maintained library costs nothing additional
in blast radius, and ``prometheus_client`` brings multiprocess-mode support and
exemplar handling that would be genuinely expensive to reimplement correctly.

Same engineer, same principle — "minimise privileged attack surface" — different
answer, because the context differs.

## Metric design

Labels are strictly bounded. ``endpoint`` uses the *route template*
(``/v1/predict``), never the raw path, because a path parameter as a label value
creates one time series per distinct value and will eventually take down the
Prometheus server. This is the single most common way people break a monitoring
stack, and it is worth guarding at the point of instrumentation.
"""

from __future__ import annotations

from collections.abc import Sequence

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from .config import ObservabilityConfig

NAMESPACE = "inference"

# Batch sizes are powers of two up to the practical maximum; this shows the
# batching engine's actual behaviour under load at a glance.
BATCH_SIZE_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)

# Queue wait is expected to be sub-millisecond when healthy, so the buckets are
# concentrated there. A spike into the high buckets means saturation.
QUEUE_WAIT_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.5, 1.0)


class ServiceMetrics:
    """All service metrics, bound to one registry.

    Taking a registry rather than using the global default means tests can
    construct an isolated instance without duplicate-timeseries errors, and the
    app can be created more than once in a single process.
    """

    def __init__(
        self,
        config: ObservabilityConfig | None = None,
        registry: CollectorRegistry | None = None,
    ) -> None:
        self._config = config or ObservabilityConfig()
        self.registry = registry if registry is not None else CollectorRegistry()

        # -- HTTP ----------------------------------------------------------
        self.http_requests_total = Counter(
            f"{NAMESPACE}_http_requests_total",
            "HTTP requests by route template, method and status class.",
            ["method", "endpoint", "status"],
            registry=self.registry,
        )
        self.http_request_duration = Histogram(
            f"{NAMESPACE}_http_request_duration_seconds",
            "End-to-end HTTP request latency, including queueing and inference.",
            ["method", "endpoint"],
            buckets=self._config.latency_buckets,
            registry=self.registry,
        )
        self.http_requests_in_flight = Gauge(
            f"{NAMESPACE}_http_requests_in_flight",
            "HTTP requests currently being served.",
            registry=self.registry,
        )

        # -- inference -----------------------------------------------------
        self.predictions_total = Counter(
            f"{NAMESPACE}_predictions_total",
            "Individual predictions returned, by predicted label.",
            ["label"],
            registry=self.registry,
        )
        self.inference_duration = Histogram(
            f"{NAMESPACE}_batch_duration_seconds",
            "Time spent inside the model's predict_batch call, per batch.",
            buckets=self._config.latency_buckets,
            registry=self.registry,
        )
        self.batch_size = Histogram(
            f"{NAMESPACE}_batch_size",
            "Number of items per inference batch. Mean near 1 means batching is "
            "not engaging; near max_batch_size means the service is saturated.",
            buckets=BATCH_SIZE_BUCKETS,
            registry=self.registry,
        )
        self.queue_wait = Histogram(
            f"{NAMESPACE}_queue_wait_seconds",
            "Time an item spent queued before its batch was dispatched.",
            buckets=QUEUE_WAIT_BUCKETS,
            registry=self.registry,
        )
        self.queue_depth = Gauge(
            f"{NAMESPACE}_queue_depth",
            "Items currently waiting in the inference queue.",
            registry=self.registry,
        )
        self.batches_total = Counter(
            f"{NAMESPACE}_batches_total",
            "Inference batches dispatched.",
            registry=self.registry,
        )
        self.rejections_total = Counter(
            f"{NAMESPACE}_rejections_total",
            "Requests rejected before inference, by reason "
            "(queue_full = backpressure, timeout = client budget exceeded).",
            ["reason"],
            registry=self.registry,
        )
        self.inference_errors_total = Counter(
            f"{NAMESPACE}_inference_errors_total",
            "Exceptions raised inside the model backend, by exception type.",
            ["error_type"],
            registry=self.registry,
        )

        # -- lifecycle -----------------------------------------------------
        self.model_info = Info(
            f"{NAMESPACE}_model",
            "Loaded model identity.",
            registry=self.registry,
        )
        self.build_info = Info(
            f"{NAMESPACE}_build",
            "Service build identity.",
            registry=self.registry,
        )
        self.model_load_duration = Gauge(
            f"{NAMESPACE}_model_load_duration_seconds",
            "Time taken to load the model artifact at startup.",
            registry=self.registry,
        )
        self.ready = Gauge(
            f"{NAMESPACE}_ready",
            "1 when the service is ready to serve traffic, 0 otherwise.",
            registry=self.registry,
        )
        self.start_time = Gauge(
            f"{NAMESPACE}_start_time_seconds",
            "Unix timestamp at which the process started serving.",
            registry=self.registry,
        )

        self.ready.set(0)

    # -- recording helpers --------------------------------------------------
    def observe_request(
        self, *, method: str, endpoint: str, status_code: int, duration_seconds: float
    ) -> None:
        # Status is recorded as a class (2xx, 4xx, 5xx) not the exact code:
        # bounded cardinality, and alerts are written against classes anyway.
        status_class = f"{status_code // 100}xx"
        self.http_requests_total.labels(method=method, endpoint=endpoint, status=status_class).inc()
        self.http_request_duration.labels(method=method, endpoint=endpoint).observe(
            duration_seconds
        )

    def observe_batch(
        self, *, size: int, duration_seconds: float, queue_times: Sequence[float]
    ) -> None:
        self.batches_total.inc()
        self.batch_size.observe(size)
        self.inference_duration.observe(duration_seconds)
        for wait in queue_times:
            self.queue_wait.observe(wait)

    def observe_predictions(self, labels: Sequence[str]) -> None:
        for label in labels:
            self.predictions_total.labels(label=label).inc()

    def record_rejection(self, reason: str) -> None:
        self.rejections_total.labels(reason=reason).inc()

    def record_inference_error(self, error_type: str) -> None:
        self.inference_errors_total.labels(error_type=error_type).inc()

    def set_queue_depth(self, depth: int) -> None:
        self.queue_depth.set(depth)

    def set_ready(self, ready: bool) -> None:
        self.ready.set(1 if ready else 0)

    def render(self) -> tuple[bytes, str]:
        """Return the exposition payload and its content type.

        Both come from the same module on purpose. ``generate_latest`` emits the
        Prometheus **text** format; pairing it with the OpenMetrics content type
        would advertise a format the body does not satisfy (OpenMetrics requires
        a trailing ``# EOF``), and a scraper honouring the header would reject
        every scrape.
        """
        return generate_latest(self.registry), CONTENT_TYPE_LATEST
