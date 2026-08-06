#!/usr/bin/env python3
"""Train the sentiment classifier and write the model artifact.

Exists so the shipped weights are **reproducible**, not magic numbers committed
by someone who no longer works here. Run it and you get a byte-identical
artifact; change the corpus or hyperparameters and you can see exactly what
moved.

    python scripts/train_model.py
    python scripts/train_model.py --epochs 400 --output models/sentiment-v2.json

Logistic regression over hashed n-grams, trained by mini-batch gradient descent
with L2 regularisation. Pure Python and stdlib only, matching the runtime
backend's dependency profile.

This is a small model on a small corpus, and it is not pretending otherwise —
see the honest accuracy report it prints at the end. The point of this
repository is the serving infrastructure; the model is deliberately swappable
(see docs/architecture.md).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from inference_service.model.linear import extract_features, sigmoid

DEFAULT_OUTPUT = REPO_ROOT / "models" / "sentiment-v1.json"
DIMENSIONS = 4096
NGRAM_RANGE = (1, 2)

# Labelled corpus. Deliberately small and readable so the training run is fast
# and the artifact is reviewable, with negation and mixed-polarity examples so
# the bigram features earn their place.
CORPUS: list[tuple[str, int]] = [
    # -- positive (1) ------------------------------------------------------
    ("this is excellent and works perfectly", 1),
    ("absolutely fantastic, exceeded my expectations", 1),
    ("great product, highly recommend it to everyone", 1),
    ("i love this, it is exactly what i needed", 1),
    ("wonderful experience, the support team was helpful", 1),
    ("fast delivery and the quality is superb", 1),
    ("brilliant design and very easy to use", 1),
    ("amazing value for the money, very happy", 1),
    ("works flawlessly, no complaints at all", 1),
    ("the best purchase i have made this year", 1),
    ("really impressed with how well it performs", 1),
    ("solid build quality and a beautiful finish", 1),
    ("outstanding service, they resolved it immediately", 1),
    ("very satisfied, would buy again without hesitation", 1),
    ("it is reliable, fast and genuinely useful", 1),
    ("delightful to use and beautifully documented", 1),
    ("saved me hours of work, thoroughly recommended", 1),
    ("clean interface and excellent performance", 1),
    ("not disappointed at all, it is superb", 1),
    ("never had a single problem with it", 1),
    ("good value and it arrived earlier than promised", 1),
    ("the setup was simple and everything just worked", 1),
    ("incredibly helpful and responsive team", 1),
    ("perfect for my needs, five stars", 1),
    ("smooth, polished and well engineered", 1),
    ("a pleasure to work with, top quality", 1),
    ("efficient, stable and well supported", 1),
    ("exceeded expectations in every way", 1),
    ("the documentation is clear and thorough", 1),
    ("fantastic support and a great community", 1),
    # -- negative (0) ------------------------------------------------------
    ("this is terrible and completely broken", 0),
    ("absolutely awful, a total waste of money", 0),
    ("poor quality, it failed within a week", 0),
    ("i hate this, it never works properly", 0),
    ("horrible experience, support was useless", 0),
    ("slow delivery and the item arrived damaged", 0),
    ("badly designed and very hard to use", 0),
    ("overpriced rubbish, very disappointed", 0),
    ("crashes constantly, nothing but problems", 0),
    ("the worst purchase i have made this year", 0),
    ("very unimpressed with how badly it performs", 0),
    ("flimsy build quality and an ugly finish", 0),
    ("appalling service, they ignored my messages", 0),
    ("not satisfied, i would never buy again", 0),
    ("it is unreliable, slow and frankly useless", 0),
    ("painful to use and the docs are missing", 0),
    ("wasted hours of my time, avoid completely", 0),
    ("cluttered interface and dreadful performance", 0),
    ("not good at all, deeply disappointing", 0),
    ("had nothing but trouble with it", 0),
    ("bad value and it arrived far too late", 0),
    ("the setup was confusing and nothing worked", 0),
    ("completely unhelpful and unresponsive team", 0),
    ("useless for my needs, one star", 0),
    ("clunky, buggy and poorly engineered", 0),
    ("a nightmare to work with, shoddy quality", 0),
    ("inefficient, unstable and abandoned", 0),
    ("failed to meet expectations in every way", 0),
    ("the documentation is confusing and incomplete", 0),
    ("terrible support and a hostile community", 0),
]

# Held out for the honesty check printed at the end. Includes negation cases
# the unigram features alone would get wrong.
HOLDOUT: list[tuple[str, int]] = [
    ("this product is excellent and i recommend it", 1),
    ("a wonderful and reliable tool", 1),
    ("very happy with the quality", 1),
    ("works perfectly every time", 1),
    ("great support and fast responses", 1),
    ("this product is terrible and i regret it", 0),
    ("an awful and unreliable tool", 0),
    ("very unhappy with the quality", 0),
    ("fails constantly every time", 0),
    ("poor support and slow responses", 0),
]


def train(
    corpus: list[tuple[str, int]],
    *,
    dimensions: int,
    ngram_range: tuple[int, int],
    epochs: int,
    learning_rate: float,
    l2: float,
    batch_size: int,
    seed: int,
) -> tuple[dict[int, float], float]:
    """Mini-batch gradient descent on binary cross-entropy."""
    rng = random.Random(seed)

    featurised = [
        (
            extract_features(text, dimensions=dimensions, ngram_range=ngram_range),
            float(label),
        )
        for text, label in corpus
    ]

    weights: dict[int, float] = {}
    bias = 0.0

    for epoch in range(epochs):
        rng.shuffle(featurised)
        epoch_loss = 0.0

        for start in range(0, len(featurised), batch_size):
            batch = featurised[start : start + batch_size]
            weight_grads: dict[int, float] = {}
            bias_grad = 0.0

            for features, target in batch:
                logit = bias + sum(
                    value * weights.get(index, 0.0) for index, value in features.items()
                )
                predicted = sigmoid(logit)
                error = predicted - target

                # Clamped to keep log() finite when the model is confident and
                # correct; log(0) is -inf and would poison the loss trace.
                probability = min(max(predicted, 1e-12), 1 - 1e-12)
                epoch_loss -= target * math.log(probability) + (1 - target) * math.log(
                    1 - probability
                )

                for index, value in features.items():
                    weight_grads[index] = weight_grads.get(index, 0.0) + error * value
                bias_grad += error

            scale = learning_rate / len(batch)
            for index, gradient in weight_grads.items():
                current = weights.get(index, 0.0)
                # L2 shrinkage applied only to features in this batch. Weights
                # are sparse, so decaying every stored weight each step would be
                # both slower and a different (stronger) regulariser.
                weights[index] = current - scale * gradient - learning_rate * l2 * current
            bias -= scale * bias_grad  # bias is conventionally not regularised

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}  loss {epoch_loss / len(featurised):.5f}")

    # Drop weights that rounding would make irrelevant. Shrinks the artifact
    # without measurably changing predictions.
    weights = {index: value for index, value in weights.items() if abs(value) > 1e-6}
    return weights, bias


def evaluate(
    samples: list[tuple[str, int]],
    weights: dict[int, float],
    bias: float,
    *,
    dimensions: int,
    ngram_range: tuple[int, int],
) -> tuple[float, list[tuple[str, int, float]]]:
    correct = 0
    misses: list[tuple[str, int, float]] = []
    for text, label in samples:
        features = extract_features(text, dimensions=dimensions, ngram_range=ngram_range)
        logit = bias + sum(value * weights.get(index, 0.0) for index, value in features.items())
        probability = sigmoid(logit)
        predicted = 1 if probability >= 0.5 else 0
        if predicted == label:
            correct += 1
        else:
            misses.append((text, label, probability))
    return correct / len(samples), misses


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dimensions", type=int, default=DIMENSIONS)
    parser.add_argument(
        "--seed",
        type=int,
        default=20260806,
        help="fixed so the artifact is byte-reproducible",
    )
    parser.add_argument("--version", default="1.0.0")
    args = parser.parse_args()

    print(f"training on {len(CORPUS)} samples, {args.epochs} epochs")
    started = time.perf_counter()
    weights, bias = train(
        CORPUS,
        dimensions=args.dimensions,
        ngram_range=NGRAM_RANGE,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    duration = time.perf_counter() - started

    train_acc, _ = evaluate(
        CORPUS, weights, bias, dimensions=args.dimensions, ngram_range=NGRAM_RANGE
    )
    holdout_acc, misses = evaluate(
        HOLDOUT, weights, bias, dimensions=args.dimensions, ngram_range=NGRAM_RANGE
    )

    print(f"\n  trained in {duration:.2f}s")
    print(f"  non-zero weights : {len(weights)}")
    print(f"  train accuracy   : {train_acc:.1%}")
    print(f"  holdout accuracy : {holdout_acc:.1%}  ({len(HOLDOUT)} samples)")
    if misses:
        print("  holdout misses:")
        for text, label, probability in misses:
            print(f"    [{label}] p={probability:.3f}  {text!r}")

    artifact = {
        "schema_version": "1.0",
        "name": "sentiment",
        "version": args.version,
        "task": "text-classification",
        "labels": ["negative", "positive"],
        "featurizer": {
            "kind": "hashed_ngrams",
            "dimensions": args.dimensions,
            "ngram_range": list(NGRAM_RANGE),
            "lowercase": True,
        },
        # Sorted by integer index so the JSON is byte-stable across runs and
        # diffs between model versions are readable.
        "weights": {str(index): round(weights[index], 8) for index in sorted(weights)},
        "bias": round(bias, 8),
        "training": {
            "samples": len(CORPUS),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "train_accuracy": round(train_acc, 4),
            "holdout_accuracy": round(holdout_acc, 4),
            "trained_by": "scripts/train_model.py",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" so the artifact is byte-identical on Windows and Linux; CI
    # can then assert reproducibility by hash.
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    size_kb = args.output.stat().st_size / 1024
    print(f"\nwrote {args.output} ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
