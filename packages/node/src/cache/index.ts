/**
 * LRU Cache with TTL support.
 *
 * Generic cache used for embedding vectors and classification results.
 * No threading lock needed in JS (single-threaded).
 */

interface CacheEntry<T> {
  value: T;
  timestamp: number;
}

export class LRUCache<T> {
  private readonly _maxSize: number;
  private readonly _ttlMs: number;
  private readonly _cache: Map<string, CacheEntry<T>>;

  /**
   * @param maxSize Maximum number of items to store.
   * @param ttlSeconds Time-to-live for each entry in seconds.
   */
  constructor(maxSize = 300, ttlSeconds = 300.0) {
    this._maxSize = maxSize;
    this._ttlMs = ttlSeconds * 1000;
    this._cache = new Map();
  }

  /** Get a value from cache. Returns undefined if missing or expired. */
  get(key: string): T | undefined {
    const entry = this._cache.get(key);
    if (!entry) return undefined;

    // Check TTL
    if (Date.now() - entry.timestamp > this._ttlMs) {
      this._cache.delete(key);
      return undefined;
    }

    // Move to end (mark as recently used) by re-inserting
    this._cache.delete(key);
    this._cache.set(key, entry);
    return entry.value;
  }

  /** Set a value in cache. */
  set(key: string, value: T): void {
    // Remove if exists (to update position)
    if (this._cache.has(key)) {
      this._cache.delete(key);
    }

    // Evict oldest if at capacity
    while (this._cache.size >= this._maxSize) {
      const firstKey = this._cache.keys().next().value;
      if (firstKey !== undefined) {
        this._cache.delete(firstKey);
      } else {
        break;
      }
    }

    this._cache.set(key, { value, timestamp: Date.now() });
  }

  /** Clear all cached entries. */
  clear(): void {
    this._cache.clear();
  }

  /** Current number of items in cache. */
  get size(): number {
    return this._cache.size;
  }

  /** Check if a key exists and is not expired. */
  has(key: string): boolean {
    return this.get(key) !== undefined;
  }
}
