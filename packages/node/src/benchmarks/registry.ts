/**
 * Extensible benchmark registry.
 *
 * Allows users to register custom benchmarks with their own training queries
 * and normalization ranges. Designed for high connectivity with external
 * benchmark-creation tools.
 */

import { readFileSync, writeFileSync } from 'node:fs';

import { BenchmarkNormalizer, NormalizationRange } from '../scoring/benchmarks.js';
import { STANDARD_BENCHMARKS } from './standard.js';

/** Complete definition of a benchmark. */
export interface BenchmarkDefinition {
  /** Benchmark name (e.g., "MMLU"). */
  name: string;

  /** Human-readable description. */
  description: string;

  /** Representative prompts for centroid generation. */
  trainingQueries: string[];

  /** Normalization range for raw scores. */
  normalization: NormalizationRange;

  /** Broad category (TECHNICAL, CREATIVE, etc.). */
  broadCategory: string;

  /** Subcategories this benchmark covers. */
  subcategories: string[];

  /** Optional metadata. */
  metadata: Record<string, unknown>;
}

/** Create a BenchmarkDefinition from a plain object (e.g. loaded from JSON). */
export function benchmarkFromDict(d: Record<string, unknown>): BenchmarkDefinition {
  const norm = (d.normalization ?? {}) as Record<string, number>;
  return {
    name: (d.name as string) ?? '',
    description: (d.description as string) ?? '',
    trainingQueries: (d.training_queries as string[]) ?? (d.trainingQueries as string[]) ?? [],
    normalization: new NormalizationRange(
      norm.min_score ?? norm.minScore ?? 0,
      norm.max_score ?? norm.maxScore ?? 100,
    ),
    broadCategory: (d.broad_category as string) ?? (d.broadCategory as string) ?? 'TECHNICAL',
    subcategories: (d.subcategories as string[]) ?? [],
    metadata: (d.metadata as Record<string, unknown>) ?? {},
  };
}

/** Serialize a BenchmarkDefinition to a plain object. */
export function benchmarkToDict(b: BenchmarkDefinition): Record<string, unknown> {
  return {
    name: b.name,
    description: b.description,
    training_queries: b.trainingQueries,
    normalization: {
      min_score: b.normalization.minScore,
      max_score: b.normalization.maxScore,
    },
    broad_category: b.broadCategory,
    subcategories: b.subcategories,
    metadata: b.metadata,
  };
}

/**
 * Registry for benchmark definitions.
 *
 * Provides a clean interface for:
 *   - Registering custom benchmarks
 *   - Loading benchmarks from JSON files (for tool connectivity)
 *   - Exporting benchmark definitions
 *   - Integrating with the centroid generator and scoring engine
 */
export class BenchmarkRegistry {
  private _benchmarks: Map<string, BenchmarkDefinition>;

  constructor() {
    this._benchmarks = new Map();
  }

  /** Create registry with the standard 12 benchmarks. */
  static default(): BenchmarkRegistry {
    const registry = new BenchmarkRegistry();
    for (const benchmark of STANDARD_BENCHMARKS) {
      registry._benchmarks.set(benchmark.name, benchmark);
    }
    return registry;
  }

  /** Register a new benchmark or update an existing one. */
  register(benchmark: BenchmarkDefinition): void {
    this._benchmarks.set(benchmark.name, benchmark);
  }

  /** Remove a benchmark. Returns true if it existed. */
  unregister(name: string): boolean {
    return this._benchmarks.delete(name);
  }

  /** Get a benchmark by name. */
  get(name: string): BenchmarkDefinition | undefined {
    return this._benchmarks.get(name);
  }

  /** All registered benchmark names. */
  get names(): string[] {
    return [...this._benchmarks.keys()];
  }

  /** All registered benchmarks. */
  get allBenchmarks(): BenchmarkDefinition[] {
    return [...this._benchmarks.values()];
  }

  /** Get all training queries grouped by benchmark name. */
  getTrainingQueries(): Record<string, string[]> {
    const result: Record<string, string[]> = {};
    for (const [name, b] of this._benchmarks) {
      if (b.trainingQueries.length > 0) {
        result[name] = b.trainingQueries;
      }
    }
    return result;
  }

  /** Create a BenchmarkNormalizer from all registered benchmarks. */
  getNormalizer(): BenchmarkNormalizer {
    const normalizer = new BenchmarkNormalizer();
    for (const [name, benchmark] of this._benchmarks) {
      normalizer.registerRange(
        name,
        benchmark.normalization.minScore,
        benchmark.normalization.maxScore,
        benchmark.description,
      );
    }
    return normalizer;
  }

  /**
   * Load benchmarks from a JSON file.
   *
   * @returns Number of benchmarks loaded.
   */
  loadFromFile(path: string): number {
    const raw = readFileSync(path, 'utf-8');
    const data = JSON.parse(raw);

    let count = 0;
    for (const item of data.benchmarks ?? []) {
      const benchmark = benchmarkFromDict(item);
      this.register(benchmark);
      count++;
    }
    return count;
  }

  /** Export all benchmarks to a JSON file. */
  exportToFile(path: string): void {
    const data = {
      benchmarks: [...this._benchmarks.values()].map(benchmarkToDict),
    };
    writeFileSync(path, JSON.stringify(data, null, 2));
  }

  get length(): number {
    return this._benchmarks.size;
  }

  has(name: string): boolean {
    return this._benchmarks.has(name);
  }
}
