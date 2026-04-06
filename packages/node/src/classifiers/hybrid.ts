/**
 * Hybrid classifier -- combines embedding and keyword classifiers.
 *
 * Uses the embedding classifier as primary, with automatic fallback
 * to the keyword classifier when confidence is too low or when the
 * embedding provider is unavailable.
 */

import { BaseClassifier, ClassificationResult } from './base.js';
import type { EmbeddingClassifier } from './embedding.js';
import { KeywordClassifier } from './keyword.js';

/**
 * Hybrid classifier with automatic fallback.
 *
 * Strategy:
 *   1. Try embedding classifier first (semantic understanding)
 *   2. If confidence < threshold -> fall back to keyword classifier
 *   3. If embedding classifier fails -> fall back to keyword classifier
 *   4. Merge results: embedding scores are primary, keyword fills gaps
 *
 * This ensures the router always returns useful results, even without
 * an embedding model (e.g., in CI environments or on first run).
 */
export class HybridClassifier extends BaseClassifier {
  private _embedding: EmbeddingClassifier | null;
  private _keyword: KeywordClassifier;
  private _threshold: number;

  constructor(
    embeddingClassifier: EmbeddingClassifier | null = null,
    keywordClassifier?: KeywordClassifier,
    confidenceThreshold = 0.05,
  ) {
    super();
    this._embedding = embeddingClassifier;
    this._keyword = keywordClassifier ?? new KeywordClassifier();
    this._threshold = confidenceThreshold;
  }

  classify(prompt: string): ClassificationResult {
    const start = performance.now();

    // Try embedding classifier first
    if (this._embedding !== null) {
      try {
        const result = this._embedding.classify(prompt);

        if (result.confidence >= this._threshold) {
          return {
            ...result,
            classifierUsed: 'hybrid(embedding)',
            processingTimeMs: performance.now() - start,
          };
        }
        // Low confidence -- fall through to keyword
      } catch {
        // Embedding classifier failed -- fall through to keyword
      }
    }

    // Fallback to keyword classifier
    const result = this._keyword.classify(prompt);
    return {
      ...result,
      classifierUsed: 'hybrid(keyword_fallback)',
      processingTimeMs: performance.now() - start,
    };
  }

  isReady(): boolean {
    // Always ready because keyword classifier needs no setup
    return true;
  }
}
