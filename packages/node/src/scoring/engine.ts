/**
 * Dynamic model scoring engine.
 *
 * Combines benchmark performance, cost, and speed into a single score
 * weighted by user priorities. This is the heart of the routing logic.
 */

import { ModelInfo } from '../registry/models.js';
import { BenchmarkNormalizer } from './benchmarks.js';
import { DEFAULT_PRIORITIES, Priorities } from './priorities.js';

export interface ModelScore {
  modelId: string;
  finalScore: number;       // 0-1 combined score
  qualityScore: number;     // 0-1 benchmark quality
  costScore: number;        // 0-1 (higher = cheaper)
  speedScore: number;       // 0-1 (higher = faster)
  qualityContribution: number;
  costContribution: number;
  speedContribution: number;
  topBenchmarks: Array<[string, number]>; // Most relevant benchmarks for this model
  reasoning: string;        // Human-readable explanation
}

/** Speed tier -> numeric score. */
export const SPEED_SCORES: Record<string, number> = {
  'very fast': 0.5,
  'fast': 0.4,
  'medium': 0.3,
  'slow': 0.2,
  'very slow': 0.1,
};

/**
 * Scores models against a classified prompt.
 *
 * Takes benchmark similarity scores (from the classifier) and user priorities,
 * then ranks all available models using a three-factor weighted algorithm:
 *
 *     final = (quality * qW + cost * cW + speed * sW) / (qW + cW + sW)
 *
 * Where weights are derived from user priorities (1-5 scale).
 */
export class ScoringEngine {
  private _normalizer: BenchmarkNormalizer;

  constructor(normalizer?: BenchmarkNormalizer) {
    this._normalizer = normalizer ?? new BenchmarkNormalizer();
  }

  /**
   * Score and rank models based on benchmark similarities and priorities.
   */
  scoreModels(
    models: ModelInfo[],
    benchmarkSimilarities: Record<string, number>,
    priorities: Priorities = DEFAULT_PRIORITIES,
    topK = 5,
  ): ModelScore[] {
    // Use top 3 most relevant benchmarks for scoring
    const sortedBenchmarks = Object.entries(benchmarkSimilarities)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);

    const topBenchmarkDict: Record<string, number> = {};
    for (const [name, score] of sortedBenchmarks) {
      topBenchmarkDict[name] = score;
    }

    const scores: ModelScore[] = [];

    for (const model of models) {
      const score = this._scoreSingleModel(model, topBenchmarkDict, priorities);
      if (score !== null) {
        scores.push(score);
      }
    }

    // Sort by final score descending
    scores.sort((a, b) => b.finalScore - a.finalScore);

    // Normalize to 0.1-0.95 range (best model ~ 0.95)
    if (scores.length > 0) {
      const maxRaw = scores[0].finalScore;
      const minRaw = scores.length > 1 ? scores[scores.length - 1].finalScore : 0;

      for (const s of scores) {
        if (maxRaw === minRaw) {
          s.finalScore = 0.5;
        } else {
          const normalized = (s.finalScore - minRaw) / (maxRaw - minRaw);
          s.finalScore = Math.round((0.1 + 0.85 * normalized) * 10000) / 10000;
        }
      }
    }

    return scores.slice(0, topK);
  }

  private _scoreSingleModel(
    model: ModelInfo,
    topBenchmarks: Record<string, number>,
    priorities: Priorities,
  ): ModelScore | null {
    // --- Quality score ---
    let weightedQualitySum = 0;
    let totalSimilarityWeight = 0;
    const modelTopBenchmarks: Array<[string, number]> = [];

    for (const [benchmarkName, userSimilarity] of Object.entries(topBenchmarks)) {
      const modelBenchScore = model.benchmarkScores[benchmarkName];
      if (modelBenchScore == null) continue;

      const normalized = this._normalizer.normalize(benchmarkName, modelBenchScore);
      weightedQualitySum += userSimilarity * normalized;
      totalSimilarityWeight += userSimilarity;
      modelTopBenchmarks.push([benchmarkName, normalized]);
    }

    if (totalSimilarityWeight === 0) return null;

    const qualityScore = weightedQualitySum / totalSimilarityWeight;

    // --- Cost score ---
    let costScore = 0;
    if (priorities.cost > 1 && model.pricing) {
      const avgCost = (model.pricing.inputPer1k + model.pricing.outputPer1k) / 2;
      // Normalize against $0.10/1k tokens baseline
      costScore = Math.max(0.0, 1.0 - avgCost / 0.1);
    }

    // --- Speed score ---
    let speedScore = 0;
    if (priorities.speed > 1 && model.latency) {
      speedScore = SPEED_SCORES[model.latency] ?? 0.3;
    }

    // --- Combine with priority weights ---
    const qWeight = priorities.qualityWeight;
    const cWeight = priorities.costWeight;
    const sWeight = priorities.speedWeight;

    const qContrib = qualityScore * qWeight;
    const cContrib = costScore * cWeight;
    const sContrib = speedScore * sWeight;

    const totalWeight = qWeight + cWeight + sWeight;
    let final = (qContrib + cContrib + sContrib) / totalWeight;
    final = Math.max(0.0, Math.min(1.0, final));

    // Generate reasoning
    const topBenchStr = modelTopBenchmarks
      .slice(0, 2)
      .map(([b, s]) => `${b} (${Math.round(s * 100)}%)`)
      .join(', ');

    let reasoning = `Quality: ${qualityScore.toFixed(2)} on [${topBenchStr}]`;
    if (costScore > 0) {
      reasoning += ` | Cost efficiency: ${costScore.toFixed(2)}`;
    }
    if (speedScore > 0) {
      reasoning += ` | Speed: ${speedScore.toFixed(2)} (${model.latency})`;
    }

    return {
      modelId: model.modelId,
      finalScore: final,
      qualityScore: Math.round(qualityScore * 10000) / 10000,
      costScore: Math.round(costScore * 10000) / 10000,
      speedScore: Math.round(speedScore * 10000) / 10000,
      qualityContribution: Math.round(qContrib * 10000) / 10000,
      costContribution: Math.round(cContrib * 10000) / 10000,
      speedContribution: Math.round(sContrib * 10000) / 10000,
      topBenchmarks: modelTopBenchmarks,
      reasoning,
    };
  }
}
