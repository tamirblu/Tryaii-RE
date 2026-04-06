"""Tests for the hybrid classifier."""

from tryaii_dre.classifiers.base import ClassificationResult
from tryaii_dre.classifiers.hybrid import HybridClassifier
from tryaii_dre.classifiers.keyword import KeywordClassifier


class MockEmbeddingClassifier:
    """Mock embedding classifier for testing fallback behavior."""

    def __init__(self, confidence: float = 0.8, should_fail: bool = False):
        self._confidence = confidence
        self._should_fail = should_fail

    def classify(self, prompt: str) -> ClassificationResult:
        if self._should_fail:
            raise RuntimeError("Embedding model not available")

        return ClassificationResult(
            benchmark_scores={"HumanEval": self._confidence, "MMLU": 0.3},
            broad_category="TECHNICAL",
            subcategory="CODE_TECHNICAL",
            confidence=self._confidence,
            classifier_used="embedding",
        )

    def is_ready(self) -> bool:
        return not self._should_fail


class TestHybridClassifier:
    def test_uses_embedding_when_confident(self):
        hybrid = HybridClassifier(
            embedding_classifier=MockEmbeddingClassifier(confidence=0.8),
            confidence_threshold=0.05,
        )
        result = hybrid.classify("Write a function")
        assert "embedding" in result.classifier_used

    def test_falls_back_to_keyword_when_low_confidence(self):
        hybrid = HybridClassifier(
            embedding_classifier=MockEmbeddingClassifier(confidence=0.01),
            confidence_threshold=0.05,
        )
        result = hybrid.classify("Write a function")
        assert "keyword" in result.classifier_used

    def test_falls_back_on_embedding_error(self):
        hybrid = HybridClassifier(
            embedding_classifier=MockEmbeddingClassifier(should_fail=True),
            confidence_threshold=0.05,
        )
        result = hybrid.classify("Write a function")
        assert "keyword" in result.classifier_used

    def test_works_without_embedding_classifier(self):
        hybrid = HybridClassifier(
            embedding_classifier=None,
        )
        result = hybrid.classify("Debug this code")
        assert "keyword" in result.classifier_used
        assert result.broad_category == "TECHNICAL"

    def test_is_always_ready(self):
        hybrid = HybridClassifier()
        assert hybrid.is_ready()
