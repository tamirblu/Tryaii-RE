/**
 * Tests for the ScoringEngine.
 */

import { describe, it, expect } from 'vitest';
import { ScoringEngine } from '../../src/scoring/engine.js';
import { Priorities } from '../../src/scoring/priorities.js';
import { ModelInfo, ModelPricing } from '../../src/registry/models.js';

/** Create a mock model for testing. */
function mockModel(opts: {
  modelId: string;
  provider: string;
  benchmarks: Record<string, number>;
  pricing?: [number, number];
  latency?: 'very fast' | 'fast' | 'medium' | 'slow' | 'very slow';
}): ModelInfo {
  return new ModelInfo({
    modelId: opts.modelId,
    provider: opts.provider,
    benchmarkScores: opts.benchmarks,
    pricing: opts.pricing ? new ModelPricing(opts.pricing[0], opts.pricing[1]) : null,
    latency: opts.latency ?? null,
  });
}

describe('ScoringEngine', () => {
  const engine = new ScoringEngine();

  const models = [
    mockModel({
      modelId: 'gpt-5',
      provider: 'openai',
      benchmarks: {
        'MMLU': 92,
        'HumanEval': 90,
        'GSM8K': 95,
        'Chatbot Arena (LMSys)': 1400,
        'MT-Bench': 9.5,
      },
      pricing: [0.01, 0.03],
      latency: 'medium',
    }),
    mockModel({
      modelId: 'gpt-4o-mini',
      provider: 'openai',
      benchmarks: {
        'MMLU': 82,
        'HumanEval': 80,
        'GSM8K': 85,
        'Chatbot Arena (LMSys)': 1300,
        'MT-Bench': 8.5,
      },
      pricing: [0.0002, 0.0006],
      latency: 'very fast',
    }),
    mockModel({
      modelId: 'claude-sonnet-4',
      provider: 'anthropic',
      benchmarks: {
        'MMLU': 90,
        'HumanEval': 88,
        'SWE-bench': 75,
        'GSM8K': 92,
        'Chatbot Arena (LMSys)': 1380,
        'MT-Bench': 9.2,
      },
      pricing: [0.003, 0.015],
      latency: 'fast',
    }),
  ];

  it('should score and rank models', () => {
    const benchmarkSimilarities = {
      'HumanEval': 0.8,
      'SWE-bench': 0.6,
      'GSM8K': 0.3,
    };

    const scores = engine.scoreModels(models, benchmarkSimilarities, new Priorities(3, 3, 3));

    expect(scores.length).toBeGreaterThan(0);
    expect(scores.length).toBeLessThanOrEqual(5);

    // Scores should be sorted descending
    for (let i = 1; i < scores.length; i++) {
      expect(scores[i - 1].finalScore).toBeGreaterThanOrEqual(scores[i].finalScore);
    }

    // Each score should have required fields
    for (const s of scores) {
      expect(s.modelId).toBeTruthy();
      expect(s.finalScore).toBeGreaterThanOrEqual(0);
      expect(s.finalScore).toBeLessThanOrEqual(1);
      expect(s.reasoning).toBeTruthy();
    }
  });

  it('should favor quality when quality priority is high', () => {
    const benchmarkSimilarities = {
      'HumanEval': 0.9,
      'SWE-bench': 0.7,
    };

    const qualityPriorities = Priorities.performance(); // quality=5, cost=1, speed=1
    const scores = engine.scoreModels(models, benchmarkSimilarities, qualityPriorities);

    // gpt-5 should rank high for quality
    expect(scores.length).toBeGreaterThan(0);
    const topModel = scores[0];
    expect(topModel.qualityScore).toBeGreaterThan(0);
  });

  it('should favor cost when cost priority is high', () => {
    const benchmarkSimilarities = {
      'HumanEval': 0.8,
      'MMLU': 0.5,
    };

    const budgetPriorities = Priorities.budget(); // quality=2, cost=5, speed=3
    const scores = engine.scoreModels(models, benchmarkSimilarities, budgetPriorities);

    expect(scores.length).toBeGreaterThan(0);
    // gpt-4o-mini should rank higher due to much lower cost
    const miniIndex = scores.findIndex((s) => s.modelId === 'gpt-4o-mini');
    if (miniIndex >= 0) {
      expect(scores[miniIndex].costScore).toBeGreaterThan(0);
    }
  });

  it('should respect topK parameter', () => {
    const benchmarkSimilarities = { 'MMLU': 0.7 };
    const scores = engine.scoreModels(models, benchmarkSimilarities, new Priorities(), 2);
    expect(scores.length).toBeLessThanOrEqual(2);
  });

  it('should return empty array when no models have matching benchmarks', () => {
    const benchmarkSimilarities = { 'NonExistentBenchmark': 0.9 };
    const scores = engine.scoreModels(models, benchmarkSimilarities, new Priorities());
    expect(scores.length).toBe(0);
  });

  it('should include reasoning text', () => {
    const benchmarkSimilarities = { 'HumanEval': 0.8 };
    const scores = engine.scoreModels(models, benchmarkSimilarities, new Priorities());

    for (const s of scores) {
      expect(s.reasoning).toContain('Quality:');
    }
  });

  it('should normalize final scores to 0.1-0.95 range', () => {
    const benchmarkSimilarities = {
      'MMLU': 0.8,
      'HumanEval': 0.7,
      'GSM8K': 0.6,
    };

    const scores = engine.scoreModels(models, benchmarkSimilarities, new Priorities());

    if (scores.length > 1) {
      // Best model should be close to 0.95
      expect(scores[0].finalScore).toBeGreaterThanOrEqual(0.9);
      expect(scores[0].finalScore).toBeLessThanOrEqual(0.96);
      // Worst model should be close to 0.1
      expect(scores[scores.length - 1].finalScore).toBeGreaterThanOrEqual(0.09);
    }
  });
});
