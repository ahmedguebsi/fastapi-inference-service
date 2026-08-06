"""Built-in text classification backend.

A hashed n-gram featuriser feeding a logistic regression, with weights loaded
from a JSON artifact. Implemented in pure Python with no third-party runtime
dependency.

**Why not ship a transformer?** The service is constrained to a container image
under 200 MB. A CPU-only PyTorch wheel is ~200 MB on its own, and ONNX Runtime
plus NumPy is ~80 MB before the model. Either blows the budget.

That constraint is worth respecting rather than arguing with, because the thing
being demonstrated here is the **serving infrastructure** — dynamic batching,
backpressure, timeouts, model lifecycle, graceful drain, observability — and all
of it is identical regardless of what sits behind
:class:`~inference_service.model.base.ModelBackend`. Swapping in ONNX is a new
class and an optional dependency group, not a re-architecture. See
``docs/architecture.md`` for the ONNX backend sketch.

The maths here is genuine, not a stub: features are hashed n-grams, scoring is a
real sparse dot product, and ``scripts/train_model.py`` reproduces the artifact
from a labelled corpus by gradient descent.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import InferenceError, ModelBackend, ModelInfo, ModelLoadError, Prediction

ARTIFACT_SCHEMA_VERSION = "1.0"

# Unicode-aware: keeps letters, digits and apostrophes so "don't" stays one
# token. Splitting on whitespace alone would keep punctuation attached and
# fragment the feature space.
_TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*", re.UNICODE)


def tokenize(text: str, *, lowercase: bool = True) -> list[str]:
    """Split text into word tokens."""
    if lowercase:
        text = text.lower()
    return _TOKEN_RE.findall(text)


def hash_feature(feature: str, dimensions: int) -> int:
    """Map a feature string to a bucket index.

    Uses BLAKE2b rather than Python's ``hash()``: the builtin is randomised per
    process via PYTHONHASHSEED, which would make the same input produce
    different predictions in different replicas of the same deployment. A stable
    hash is a correctness requirement, not a preference.
    """
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimensions


def extract_features(
    text: str, *, dimensions: int, ngram_range: tuple[int, int], lowercase: bool = True
) -> dict[int, float]:
    """Hashed n-gram bag of words with sublinear term-frequency scaling.

    Returns a sparse mapping of bucket index to weight. Sublinear scaling
    (``1 + log(count)``) stops a word repeated twenty times from dominating the
    score twenty-fold, which matters for the short user-supplied text this model
    sees.
    """
    tokens = tokenize(text, lowercase=lowercase)
    if not tokens:
        return {}

    counts: dict[int, int] = {}
    low, high = ngram_range
    for n in range(low, high + 1):
        if n <= 0 or n > len(tokens):
            continue
        for start in range(len(tokens) - n + 1):
            gram = " ".join(tokens[start : start + n])
            index = hash_feature(f"{n}:{gram}", dimensions)
            counts[index] = counts.get(index, 0) + 1

    features = {index: 1.0 + math.log(count) for index, count in counts.items()}

    # L2 normalise so a long document and a short one are scored on the same
    # scale; without it, score magnitude tracks input length.
    norm = math.sqrt(sum(value * value for value in features.values()))
    if norm > 0:
        features = {index: value / norm for index, value in features.items()}
    return features


def sigmoid(x: float) -> float:
    """Numerically stable logistic function.

    The naive ``1/(1+exp(-x))`` overflows for x around -750. Branching on the
    sign keeps the exponent negative in both cases.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class LinearTextClassifier(ModelBackend):
    """Logistic regression over hashed n-gram features."""

    BACKEND_NAME = "linear"

    def __init__(self, artifact_path: str | Path) -> None:
        self._artifact_path = Path(artifact_path)
        self._weights: dict[int, float] = {}
        self._bias: float = 0.0
        self._dimensions: int = 0
        self._ngram_range: tuple[int, int] = (1, 1)
        self._lowercase: bool = True
        self._labels: tuple[str, ...] = ()
        self._info: ModelInfo | None = None
        self._loaded = False

    @property
    def info(self) -> ModelInfo:
        if self._info is None:
            raise InferenceError("model has not been loaded; call load() first")
        return self._info

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Read and validate the artifact.

        Validation is strict: a malformed artifact must fail at startup, loudly,
        rather than produce silently wrong predictions in production.
        """
        if not self._artifact_path.is_file():
            raise ModelLoadError(f"model artifact not found: {self._artifact_path}")

        try:
            raw_bytes = self._artifact_path.read_bytes()
        except OSError as exc:
            raise ModelLoadError(f"cannot read {self._artifact_path}: {exc}") from exc

        try:
            payload: dict[str, Any] = json.loads(raw_bytes)
        except json.JSONDecodeError as exc:
            raise ModelLoadError(f"{self._artifact_path} is not valid JSON: {exc}") from exc

        schema_version = payload.get("schema_version")
        if schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ModelLoadError(
                f"unsupported artifact schema_version {schema_version!r}; "
                f"this build supports {ARTIFACT_SCHEMA_VERSION!r}"
            )

        labels = payload.get("labels")
        if not isinstance(labels, list) or len(labels) != 2:
            raise ModelLoadError(
                "artifact 'labels' must be a list of exactly two strings "
                "(binary logistic regression)"
            )
        if not all(isinstance(label, str) and label for label in labels):
            raise ModelLoadError("artifact 'labels' entries must be non-empty strings")

        featurizer = payload.get("featurizer")
        if not isinstance(featurizer, dict):
            raise ModelLoadError("artifact 'featurizer' must be an object")

        kind = featurizer.get("kind")
        if kind != "hashed_ngrams":
            raise ModelLoadError(
                f"unsupported featurizer kind {kind!r}; this backend implements 'hashed_ngrams'"
            )

        dimensions = featurizer.get("dimensions")
        if not isinstance(dimensions, int) or dimensions <= 0:
            raise ModelLoadError("featurizer 'dimensions' must be a positive integer")

        ngram_range = featurizer.get("ngram_range", [1, 1])
        if (
            not isinstance(ngram_range, list)
            or len(ngram_range) != 2
            or not all(isinstance(n, int) for n in ngram_range)
            or not 1 <= ngram_range[0] <= ngram_range[1]
        ):
            raise ModelLoadError(
                "featurizer 'ngram_range' must be [low, high] with 1 <= low <= high"
            )

        raw_weights = payload.get("weights")
        if not isinstance(raw_weights, dict):
            raise ModelLoadError("artifact 'weights' must be an object of index -> float")

        weights: dict[int, float] = {}
        for key, value in raw_weights.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise ModelLoadError(f"weight key {key!r} is not an integer index") from exc
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ModelLoadError(f"weight for index {index} is not a number")
            if not 0 <= index < dimensions:
                raise ModelLoadError(
                    f"weight index {index} is outside the featurizer dimension {dimensions}"
                )
            weights[index] = float(value)

        bias = payload.get("bias", 0.0)
        if not isinstance(bias, (int, float)) or isinstance(bias, bool):
            raise ModelLoadError("artifact 'bias' must be a number")

        self._weights = weights
        self._bias = float(bias)
        self._dimensions = dimensions
        self._ngram_range = (ngram_range[0], ngram_range[1])
        self._lowercase = bool(featurizer.get("lowercase", True))
        self._labels = tuple(labels)

        self._info = ModelInfo(
            name=str(payload.get("name", "unknown")),
            version=str(payload.get("version", "0.0.0")),
            task=str(payload.get("task", "text-classification")),
            labels=self._labels,
            backend=self.BACKEND_NAME,
            artifact_path=str(self._artifact_path),
            # Pinning the artifact hash lets you prove which weights served a
            # given prediction when someone disputes an output months later.
            artifact_sha256=hashlib.sha256(raw_bytes).hexdigest(),
            parameters=len(weights) + 1,
            metadata={
                "dimensions": dimensions,
                "ngram_range": list(self._ngram_range),
                "lowercase": self._lowercase,
                "training": payload.get("training", {}),
            },
        )
        self._loaded = True

    def predict_batch(self, inputs: Sequence[str]) -> list[Prediction]:
        if not self._loaded:
            raise InferenceError("model is not loaded")

        results: list[Prediction] = []
        negative_label, positive_label = self._labels

        for text in inputs:
            if not isinstance(text, str):
                raise InferenceError(f"expected str input, got {type(text).__name__}")

            features = extract_features(
                text,
                dimensions=self._dimensions,
                ngram_range=self._ngram_range,
                lowercase=self._lowercase,
            )
            # Sparse dot product: iterate the input's features, not the weight
            # vector. Cost is O(tokens), independent of model dimension.
            logit = self._bias + sum(
                value * self._weights.get(index, 0.0) for index, value in features.items()
            )
            positive = sigmoid(logit)
            negative = 1.0 - positive

            scores = {
                negative_label: round(negative, 6),
                positive_label: round(positive, 6),
            }
            if positive >= 0.5:
                label, score = positive_label, positive
            else:
                label, score = negative_label, negative
            results.append(Prediction(label=label, score=round(score, 6), scores=scores))

        return results

    def close(self) -> None:
        self._weights = {}
        self._loaded = False
