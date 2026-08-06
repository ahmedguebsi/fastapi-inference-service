"""Model backend abstraction.

The serving layer — batching, backpressure, timeouts, metrics, lifecycle — is
identical whether the model behind it is a linear classifier, an ONNX graph, or
a remote GPU worker. Isolating the backend behind one small protocol is what
makes that true, and it is the reason this service can be tested exhaustively
without shipping a 2 GB runtime into CI.

A backend must:

* be **synchronous**. Inference is CPU-bound; the serving layer runs it in a
  worker thread so the event loop stays responsive. A backend that did its own
  async I/O would fight that design.
* implement ``predict_batch``, not ``predict``. Single-item prediction is the
  degenerate case of a batch. Making the batch the primitive means a backend
  that can genuinely vectorise (ONNX, torch) does so naturally, rather than
  having batching bolted on later.
* be **thread-safe for reads** after ``load()``. Multiple worker threads call
  ``predict_batch`` concurrently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


class ModelError(RuntimeError):
    """Base class for model failures."""


class ModelLoadError(ModelError):
    """Raised when a model artifact cannot be loaded.

    Fatal at startup by design: a service that boots with no model and starts
    returning errors is harder to diagnose than one that refuses to boot.
    """


class InferenceError(ModelError):
    """Raised when inference fails on an otherwise valid input."""


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Identity of the loaded model. Surfaced on /v1/meta and in metrics."""

    name: str
    version: str
    task: str
    labels: tuple[str, ...]
    backend: str
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    parameters: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "task": self.task,
            "labels": list(self.labels),
            "backend": self.backend,
            "artifact_sha256": self.artifact_sha256,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class Prediction:
    """One inference result."""

    label: str
    score: float
    scores: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "score": self.score, "scores": self.scores}


class ModelBackend(ABC):
    """Contract every model implementation satisfies."""

    @property
    @abstractmethod
    def info(self) -> ModelInfo:
        """Identity of the loaded model. Only valid after ``load()``."""

    @abstractmethod
    def load(self) -> None:
        """Read the artifact and prepare for inference.

        Called once during application startup, before the service reports
        ready. Must raise :class:`ModelLoadError` on failure.
        """

    @abstractmethod
    def predict_batch(self, inputs: Sequence[str]) -> list[Prediction]:
        """Run inference over a batch.

        Must return exactly one prediction per input, in the same order. The
        serving layer relies on positional correspondence to resolve the right
        caller's future — returning a different length is a contract violation
        and is checked at runtime.
        """

    def warmup(self) -> None:
        """Run a throwaway inference to fault in lazy initialisation.

        Without this, the first real request pays for one-off costs — JIT,
        memory arena allocation, lazy imports — and shows up as a latency
        outlier immediately after every deploy. Overridable; the default runs a
        single trivial batch.
        """
        self.predict_batch(["warmup"])

    def close(self) -> None:
        """Release resources. Called during graceful shutdown."""
        return
