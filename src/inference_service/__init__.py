"""Production-grade FastAPI text classification inference service.

Dynamic request batching, bounded queueing with backpressure, per-request
timeouts, Prometheus instrumentation, structured logging with request
correlation, and a graceful shutdown that drains in-flight work.
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = ["__version__"]
