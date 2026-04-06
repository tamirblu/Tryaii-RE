"""
Hybrid classifier -- combines embedding and keyword classifiers.

Uses the embedding classifier as primary, with automatic fallback
to the keyword classifier when confidence is too low or when the
embedding provider is unavailable.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from tryaii_dre.classifiers.base import BaseClassifier, ClassificationResult
from tryaii_dre.classifiers.embedding import EmbeddingClassifier
from tryaii_dre.classifiers.keyword import KeywordClassifier

logger = logging.getLogger("tryaii_dre.classifiers")


class HybridClassifier(BaseClassifier):
    """
    Hybrid classifier with automatic fallback.

    Strategy:
        1. Try embedding classifier first (semantic understanding)
        2. If confidence < threshold -> fall back to keyword classifier
        3. If embedding classifier fails -> fall back to keyword classifier
        4. Merge results: embedding scores are primary, keyword fills gaps

    This ensures the router always returns useful results, even without
    an embedding model (e.g., in CI environments or on first run).
    """

    def __init__(
        self,
        embedding_classifier: Optional[EmbeddingClassifier] = None,
        keyword_classifier: Optional[KeywordClassifier] = None,
        confidence_threshold: float = 0.05,
    ):
        self._embedding = embedding_classifier
        self._keyword = keyword_classifier or KeywordClassifier()
        self._threshold = confidence_threshold

    def classify(self, prompt: str) -> ClassificationResult:
        """Classify using embedding-first with keyword fallback."""
        start = time.time()

        # Try embedding classifier first
        if self._embedding is not None:
            try:
                result = self._embedding.classify(prompt)

                if result.confidence >= self._threshold:
                    result.classifier_used = "hybrid(embedding)"
                    result.processing_time_ms = (time.time() - start) * 1000
                    return result
                else:
                    logger.debug(
                        f"Embedding confidence {result.confidence:.3f} "
                        f"below threshold {self._threshold}, falling back to keyword"
                    )
            except Exception as e:
                logger.warning(f"Embedding classifier failed: {e}, using keyword fallback")

        # Fallback to keyword classifier
        result = self._keyword.classify(prompt)
        result.classifier_used = "hybrid(keyword_fallback)"
        result.processing_time_ms = (time.time() - start) * 1000
        return result

    def is_ready(self) -> bool:
        # Always ready because keyword classifier needs no setup
        return True
