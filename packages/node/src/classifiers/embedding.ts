/**
 * Neural embedding classifier.
 *
 * Classifies prompts by computing cosine similarity between the prompt's
 * embedding vector and pre-computed benchmark centroids. This gives a
 * semantic understanding of "what kind of task" a prompt represents.
 */

import { createHash } from 'node:crypto';

import { LRUCache } from '../cache/index.js';
import { BaseClassifier, ClassificationResult } from './base.js';
import { cosineSimilarity } from '../utils/cosine.js';
import { BaseEmbeddingProvider } from '../embeddings/base.js';
import { CentroidLoader } from '../centroids/loader.js';

/** Benchmark -> broad category mapping for display purposes. */
export const BENCHMARK_CATEGORIES: Record<string, [string, string]> = {
  'MMLU': ['EDUCATIONAL', 'ACADEMIC_INSTRUCTION'],
  'HellaSwag': ['CONVERSATIONAL', 'PERSONAL_ADVICE'],
  'HumanEval': ['TECHNICAL', 'CODE_TECHNICAL'],
  'SWE-bench': ['TECHNICAL', 'CODE_TECHNICAL'],
  'TruthfulQA': ['CONVERSATIONAL', 'PERSONAL_ADVICE'],
  'ARC': ['EDUCATIONAL', 'ACADEMIC_INSTRUCTION'],
  'GSM8K': ['TECHNICAL', 'MATHEMATICAL_SCIENTIFIC'],
  'DROP': ['TECHNICAL', 'MATHEMATICAL_SCIENTIFIC'],
  'SuperGLUE': ['BUSINESS', 'PROFESSIONAL_COMMUNICATION'],
  'Chatbot Arena (LMSys)': ['CONVERSATIONAL', 'PERSONAL_ADVICE'],
  'MT-Bench': ['CREATIVE', 'WRITING_LITERARY'],
  'LiveBench': ['TECHNICAL', 'CODE_TECHNICAL'],
};

/**
 * Semantic classifier using embedding cosine similarity.
 *
 * Flow:
 *   1. Embed the user prompt using the configured embedding provider
 *   2. Compute cosine similarity against each benchmark centroid
 *   3. Return similarity scores as the classification result
 *
 * Includes LRU caching for both embeddings and full classification results.
 */
export class EmbeddingClassifier extends BaseClassifier {
  private _provider: BaseEmbeddingProvider;
  private _centroidLoader: CentroidLoader;
  private _embeddingCache: LRUCache<number[]>;
  private _classificationCache: LRUCache<ClassificationResult>;

  constructor(
    embeddingProvider: BaseEmbeddingProvider,
    centroidLoader: CentroidLoader,
    opts?: {
      embeddingCacheSize?: number;
      classificationCacheSize?: number;
      ttlSeconds?: number;
    },
  ) {
    super();
    this._provider = embeddingProvider;
    this._centroidLoader = centroidLoader;

    this._embeddingCache = new LRUCache<number[]>(
      opts?.embeddingCacheSize ?? 300,
      opts?.ttlSeconds ?? 300,
    );
    this._classificationCache = new LRUCache<ClassificationResult>(
      opts?.classificationCacheSize ?? 150,
      opts?.ttlSeconds ?? 300,
    );
  }

  classify(prompt: string): ClassificationResult {
    const start = performance.now();

    // Check classification cache
    const cacheKey = EmbeddingClassifier._hashPrompt(prompt);
    const cached = this._classificationCache.get(cacheKey);
    if (cached !== undefined) {
      return {
        ...cached,
        cacheHit: true,
        processingTimeMs: performance.now() - start,
      };
    }

    // Ensure centroids are loaded
    const centroids = this._centroidLoader.getCentroids();

    // Get prompt embedding (with caching)
    const embedding = this._getEmbedding(prompt);

    // Calculate cosine similarity against each benchmark centroid
    const benchmarkScores: Record<string, number> = {};
    for (const [benchmarkName, centroid] of Object.entries(centroids)) {
      const similarity = cosineSimilarity(embedding, centroid);
      // Clamp to [0, 1] -- negative similarities are not meaningful here
      benchmarkScores[benchmarkName] = Math.max(0.0, similarity);
    }

    // Determine top category from highest-scoring benchmark
    let topBenchmark = '';
    let topScore = -1;
    for (const [name, score] of Object.entries(benchmarkScores)) {
      if (score > topScore) {
        topScore = score;
        topBenchmark = name;
      }
    }

    const categories = BENCHMARK_CATEGORIES[topBenchmark] ?? ['TECHNICAL', 'CODE_TECHNICAL'];

    const result: ClassificationResult = {
      benchmarkScores,
      broadCategory: categories[0],
      subcategory: categories[1],
      confidence: topScore,
      classifierUsed: 'embedding',
      cacheHit: false,
      processingTimeMs: performance.now() - start,
    };

    // Cache the result
    this._classificationCache.set(cacheKey, result);

    return result;
  }

  private _getEmbedding(text: string): number[] {
    const cacheKey = EmbeddingClassifier._hashPrompt(text);
    const cached = this._embeddingCache.get(cacheKey);
    if (cached !== undefined) return cached;

    const embedding = this._provider.embed(text);
    this._embeddingCache.set(cacheKey, embedding);
    return embedding;
  }

  isReady(): boolean {
    return true; // Lazy initialization handles readiness
  }

  private static _hashPrompt(prompt: string): string {
    return createHash('md5').update(prompt).digest('hex');
  }
}
