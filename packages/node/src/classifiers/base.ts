/**
 * Abstract base classifier.
 *
 * All classifiers (embedding, keyword, hybrid) implement this interface,
 * so they can be swapped transparently.
 */

/** Output of any classifier. */
export interface ClassificationResult {
  /** Cosine similarity to each benchmark centroid (0-1). */
  benchmarkScores: Record<string, number>;

  /** Broad category (for display / debugging). */
  broadCategory: string;

  /** Subcategory refinement. */
  subcategory: string;

  /** Confidence of the classification (0-1). */
  confidence: number;

  /** Which classifier produced this result ("embedding", "keyword", "hybrid"). */
  classifierUsed: string;

  /** Whether this result came from cache. */
  cacheHit: boolean;

  /** How long classification took in milliseconds. */
  processingTimeMs: number;
}

/** Create a default (empty) ClassificationResult. */
export function emptyClassificationResult(): ClassificationResult {
  return {
    benchmarkScores: {},
    broadCategory: '',
    subcategory: '',
    confidence: 0,
    classifierUsed: '',
    cacheHit: false,
    processingTimeMs: 0,
  };
}

/** Get top benchmarks sorted by similarity score (descending). */
export function topBenchmarks(result: ClassificationResult): Array<[string, number]> {
  return Object.entries(result.benchmarkScores)
    .sort((a, b) => b[1] - a[1]);
}

/**
 * Abstract base class for prompt classifiers.
 *
 * A classifier takes a user prompt and returns benchmark similarity scores.
 * These scores tell us "what kind of task is this?" in terms of which
 * AI benchmarks it most resembles.
 */
export abstract class BaseClassifier {
  /**
   * Classify a prompt and return benchmark similarity scores.
   *
   * @param prompt - The user's input text.
   * @returns ClassificationResult with benchmarkScores populated.
   */
  abstract classify(prompt: string): ClassificationResult;

  /** Check if the classifier is initialized and ready to use. */
  abstract isReady(): boolean;
}
