/**
 * Cosine similarity between two vectors.
 */

import { dotProduct, vectorNorm } from './math.js';

/**
 * Compute cosine similarity between two vectors.
 *
 * @returns A value between -1 and 1 (1 = identical direction, 0 = orthogonal).
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  const dot = dotProduct(a, b);
  const normA = vectorNorm(a);
  const normB = vectorNorm(b);
  if (normA === 0 || normB === 0) return 0.0;
  return dot / (normA * normB);
}
