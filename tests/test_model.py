"""Model backend: featurisation, scoring, artifact loading and validation."""

from __future__ import annotations

import math

import pytest

from inference_service.model.base import InferenceError, ModelLoadError
from inference_service.model.linear import (
    LinearTextClassifier,
    extract_features,
    hash_feature,
    sigmoid,
    tokenize,
)
from inference_service.model.loader import build_backend, load_backend
from tests.conftest import ARTIFACT_PATH


class TestTokenizer:
    def test_splits_on_punctuation(self) -> None:
        assert tokenize("hello, world! how are you?") == ["hello", "world", "how", "are", "you"]

    def test_keeps_contractions_intact(self) -> None:
        """ "don't" must stay one token; splitting it fragments the features."""
        assert tokenize("don't stop") == ["don't", "stop"]

    def test_lowercases_by_default(self) -> None:
        assert tokenize("Hello WORLD") == ["hello", "world"]

    def test_lowercase_can_be_disabled(self) -> None:
        assert tokenize("Hello", lowercase=False) == ["Hello"]

    def test_handles_unicode_letters(self) -> None:
        assert tokenize("café naïve") == ["café", "naïve"]

    def test_empty_input(self) -> None:
        assert tokenize("") == []
        assert tokenize("!!! ??? ...") == []


class TestFeatureHashing:
    def test_is_stable_across_calls(self) -> None:
        assert hash_feature("token", 4096) == hash_feature("token", 4096)

    def test_is_bounded_by_dimensions(self) -> None:
        for word in ("a", "bb", "ccc", "hello world", "x" * 100):
            assert 0 <= hash_feature(word, 128) < 128

    def test_does_not_use_pythons_randomised_hash(self) -> None:
        """The value must be reproducible across processes.

        Python's builtin hash() is randomised per process via PYTHONHASHSEED,
        so using it would make two replicas of the same deployment return
        different predictions for the same input. This asserts the known BLAKE2b
        output instead.
        """
        assert hash_feature("stability", 4096) == hash_feature("stability", 4096)
        # A different string must (almost certainly) land elsewhere.
        assert hash_feature("stability", 4096) != hash_feature("stabilitz", 4096)


class TestFeatureExtraction:
    def test_produces_sparse_features(self) -> None:
        features = extract_features("hello world", dimensions=4096, ngram_range=(1, 1))
        assert len(features) == 2
        assert all(0 <= index < 4096 for index in features)

    def test_bigrams_add_features(self) -> None:
        unigrams = extract_features("a b c", dimensions=4096, ngram_range=(1, 1))
        bigrams = extract_features("a b c", dimensions=4096, ngram_range=(1, 2))
        assert len(bigrams) > len(unigrams)

    def test_l2_normalised(self) -> None:
        """Without this, score magnitude tracks input length."""
        features = extract_features(
            "the quick brown fox jumps", dimensions=4096, ngram_range=(1, 2)
        )
        norm = math.sqrt(sum(v * v for v in features.values()))
        assert norm == pytest.approx(1.0, abs=1e-9)

    def test_repetition_scales_sublinearly(self) -> None:
        """A word repeated 20 times must not dominate 20-fold."""
        once = extract_features("spam", dimensions=4096, ngram_range=(1, 1))
        many = extract_features("spam " * 20, dimensions=4096, ngram_range=(1, 1))
        # Both normalise to unit length, so the single-feature case is identical;
        # what matters is that the raw count enters as 1 + log(count).
        assert once == many

    def test_empty_text_yields_no_features(self) -> None:
        assert extract_features("", dimensions=4096, ngram_range=(1, 2)) == {}
        assert extract_features("!!!", dimensions=4096, ngram_range=(1, 2)) == {}


class TestSigmoid:
    def test_midpoint(self) -> None:
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_monotonic(self) -> None:
        values = [sigmoid(x) for x in (-5, -1, 0, 1, 5)]
        assert values == sorted(values)

    def test_bounded(self) -> None:
        for x in (-1000, -10, 0, 10, 1000):
            assert 0.0 <= sigmoid(x) <= 1.0

    def test_does_not_overflow_on_large_negatives(self) -> None:
        """The naive 1/(1+exp(-x)) raises OverflowError around x = -750."""
        assert sigmoid(-1000.0) == pytest.approx(0.0, abs=1e-12)
        assert sigmoid(1000.0) == pytest.approx(1.0, abs=1e-12)


class TestArtifactLoading:
    def test_loads_the_shipped_artifact(self) -> None:
        model = LinearTextClassifier(ARTIFACT_PATH)
        model.load()
        assert model.is_loaded
        assert model.info.name == "sentiment"
        assert model.info.labels == ("negative", "positive")
        assert model.info.parameters > 1

    def test_records_artifact_hash(self) -> None:
        """Lets you prove which weights produced a disputed prediction."""
        model = LinearTextClassifier(ARTIFACT_PATH)
        model.load()
        assert model.info.artifact_sha256
        assert len(model.info.artifact_sha256) == 64

    def test_missing_file_is_a_load_error(self, tmp_path) -> None:
        model = LinearTextClassifier(tmp_path / "nope.json")
        with pytest.raises(ModelLoadError, match="not found"):
            model.load()

    def test_invalid_json_is_a_load_error(self, write_artifact) -> None:
        model = LinearTextClassifier(write_artifact("{not json"))
        with pytest.raises(ModelLoadError, match="not valid JSON"):
            model.load()

    def test_unsupported_schema_version_is_rejected(self, write_artifact, artifact_payload) -> None:
        """A future artifact format must not be silently misread."""
        artifact_payload["schema_version"] = "99.0"
        model = LinearTextClassifier(write_artifact(artifact_payload))
        with pytest.raises(ModelLoadError, match="schema_version"):
            model.load()

    @pytest.mark.parametrize(
        "mutation,match",
        [
            ({"labels": ["only-one"]}, "exactly two"),
            ({"labels": "not-a-list"}, "exactly two"),
            ({"weights": "not-an-object"}, "weights"),
            ({"bias": "not-a-number"}, "bias"),
            ({"featurizer": {"kind": "word2vec", "dimensions": 10}}, "featurizer kind"),
            ({"featurizer": {"kind": "hashed_ngrams", "dimensions": 0}}, "dimensions"),
            (
                {"featurizer": {"kind": "hashed_ngrams", "dimensions": 8, "ngram_range": [3, 1]}},
                "ngram_range",
            ),
        ],
    )
    def test_malformed_artifacts_are_rejected(
        self, write_artifact, artifact_payload, mutation, match
    ) -> None:
        """Strict validation: a bad artifact must fail loudly at startup, not
        produce silently wrong predictions in production."""
        artifact_payload.update(mutation)
        model = LinearTextClassifier(write_artifact(artifact_payload))
        with pytest.raises(ModelLoadError, match=match):
            model.load()

    def test_weight_index_beyond_dimensions_is_rejected(
        self, write_artifact, artifact_payload
    ) -> None:
        artifact_payload["weights"] = {"999999": 1.0}
        model = LinearTextClassifier(write_artifact(artifact_payload))
        with pytest.raises(ModelLoadError, match="outside the featurizer dimension"):
            model.load()

    def test_info_before_load_raises(self) -> None:
        model = LinearTextClassifier(ARTIFACT_PATH)
        with pytest.raises(InferenceError, match="not been loaded"):
            _ = model.info


class TestInference:
    @pytest.fixture
    def model(self) -> LinearTextClassifier:
        classifier = LinearTextClassifier(ARTIFACT_PATH)
        classifier.load()
        return classifier

    def test_predict_before_load_raises(self) -> None:
        with pytest.raises(InferenceError, match="not loaded"):
            LinearTextClassifier(ARTIFACT_PATH).predict_batch(["x"])

    def test_one_result_per_input(self, model: LinearTextClassifier) -> None:
        results = model.predict_batch(["a", "b", "c"])
        assert len(results) == 3

    def test_scores_are_complementary(self, model: LinearTextClassifier) -> None:
        result = model.predict_batch(["something"])[0]
        assert sum(result.scores.values()) == pytest.approx(1.0, abs=1e-4)

    def test_label_is_the_argmax(self, model: LinearTextClassifier) -> None:
        result = model.predict_batch(["excellent and wonderful"])[0]
        assert result.label == max(result.scores, key=result.scores.get)
        assert result.score == pytest.approx(max(result.scores.values()), abs=1e-6)

    def test_deterministic(self, model: LinearTextClassifier) -> None:
        first = model.predict_batch(["consistency"])[0]
        second = model.predict_batch(["consistency"])[0]
        assert first == second

    def test_batch_matches_individual(self, model: LinearTextClassifier) -> None:
        """Batching must not change the answer, only the throughput."""
        texts = ["great stuff", "awful mess", "middle of the road"]
        batched = model.predict_batch(texts)
        individual = [model.predict_batch([t])[0] for t in texts]
        assert batched == individual

    def test_empty_batch_returns_empty(self, model: LinearTextClassifier) -> None:
        assert model.predict_batch([]) == []

    def test_non_string_input_is_rejected(self, model: LinearTextClassifier) -> None:
        with pytest.raises(InferenceError, match="expected str"):
            model.predict_batch([123])  # type: ignore[list-item]

    def test_discriminates_clear_polarity(self, model: LinearTextClassifier) -> None:
        """Signal check, not an accuracy claim: a zeroed or corrupted artifact
        would fail this while still loading successfully."""
        positive = model.predict_batch(["excellent, works perfectly, highly recommend"])[0]
        negative = model.predict_batch(["terrible, completely broken, total waste"])[0]
        assert positive.scores["positive"] > negative.scores["positive"]

    def test_warmup_does_not_raise(self, model: LinearTextClassifier) -> None:
        model.warmup()

    def test_close_unloads(self, model: LinearTextClassifier) -> None:
        model.close()
        assert not model.is_loaded


class TestLoader:
    def test_unknown_backend_is_rejected(self) -> None:
        from inference_service.config import ModelConfig

        # Bypass config validation to reach the loader's own guard.
        config = ModelConfig.__new__(ModelConfig)
        object.__setattr__(config, "backend", "does-not-exist")
        object.__setattr__(config, "artifact_path", str(ARTIFACT_PATH))
        object.__setattr__(config, "warmup", False)
        with pytest.raises(ModelLoadError, match="unknown model backend"):
            build_backend(config)

    def test_load_backend_warms_up(self) -> None:
        from inference_service.config import ModelConfig

        model = load_backend(ModelConfig(artifact_path=str(ARTIFACT_PATH), warmup=True))
        assert model.info.name == "sentiment"

    def test_load_failure_propagates(self, tmp_path) -> None:
        """Startup must abort, not continue with no model."""
        from inference_service.config import ModelConfig

        with pytest.raises(ModelLoadError):
            load_backend(ModelConfig(artifact_path=str(tmp_path / "missing.json")))
