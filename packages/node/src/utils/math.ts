/**
 * Basic vector math utilities.
 *
 * Replaces numpy operations with plain TypeScript array math.
 */

/** Compute the dot product of two vectors. */
export function dotProduct(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

/** Compute the L2 (Euclidean) norm of a vector. */
export function vectorNorm(v: number[]): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    sum += v[i] * v[i];
  }
  return Math.sqrt(sum);
}

/** Compute the element-wise mean of a list of vectors. */
export function vectorMean(vectors: number[][]): number[] {
  if (vectors.length === 0) return [];
  const dim = vectors[0].length;
  const result = new Array<number>(dim).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < dim; i++) {
      result[i] += v[i];
    }
  }
  const n = vectors.length;
  for (let i = 0; i < dim; i++) {
    result[i] /= n;
  }
  return result;
}

/** Scale a vector by a scalar. */
export function vectorScale(v: number[], scalar: number): number[] {
  return v.map((x) => x * scalar);
}

/** Add two vectors element-wise. */
export function vectorAdd(a: number[], b: number[]): number[] {
  return a.map((x, i) => x + b[i]);
}

/** Normalize a vector to unit length. Returns zero vector if norm is 0. */
export function vectorNormalize(v: number[]): number[] {
  const norm = vectorNorm(v);
  if (norm === 0) return v.slice();
  return v.map((x) => x / norm);
}
