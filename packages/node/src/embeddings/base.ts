/**
 * Abstract embedding provider interface.
 *
 * Allows swapping between local (@xenova/transformers) and cloud (OpenAI)
 * embedding backends without changing the classifier logic.
 */

/**
 * Base class for embedding providers.
 *
 * All providers must implement embed(), embedBatch(), dimension, and modelName.
 */
export abstract class BaseEmbeddingProvider {
  /**
   * Generate an embedding vector for a single text.
   *
   * @param text - Input text to embed.
   * @returns A 1-D array of numbers (the embedding vector).
   */
  abstract embed(text: string): number[];

  /**
   * Generate embeddings for multiple texts efficiently.
   *
   * @param texts - List of input texts.
   * @returns List of 1-D number arrays.
   */
  abstract embedBatch(texts: string[]): number[][];

  /** Dimensionality of the embedding vectors. */
  abstract get dimension(): number;

  /** Name/identifier of the embedding model. */
  abstract get modelName(): string;
}
