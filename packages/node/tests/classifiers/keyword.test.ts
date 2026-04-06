/**
 * Tests for the KeywordClassifier.
 */

import { describe, it, expect } from 'vitest';
import { KeywordClassifier } from '../../src/classifiers/keyword.js';

describe('KeywordClassifier', () => {
  const classifier = new KeywordClassifier();

  it('should always be ready', () => {
    expect(classifier.isReady()).toBe(true);
  });

  it('should classify a coding prompt as TECHNICAL', () => {
    const result = classifier.classify('Write a Python function to sort an array using quicksort algorithm');
    expect(result.broadCategory).toBe('TECHNICAL');
    expect(result.classifierUsed).toBe('keyword');
    expect(result.cacheHit).toBe(false);
    expect(result.processingTimeMs).toBeGreaterThanOrEqual(0);
  });

  it('should classify a coding prompt with CODE_TECHNICAL subcategory', () => {
    const result = classifier.classify('Debug this Python function that uses a class method');
    expect(result.broadCategory).toBe('TECHNICAL');
    expect(result.subcategory).toBe('CODE_TECHNICAL');
    // Should have HumanEval or SWE-bench in benchmark scores
    expect(result.benchmarkScores['HumanEval']).toBeGreaterThan(0);
  });

  it('should classify a creative writing prompt as CREATIVE', () => {
    const result = classifier.classify('Write a short story about a dragon who learns to paint artwork');
    expect(result.broadCategory).toBe('CREATIVE');
    expect(result.benchmarkScores['MT-Bench']).toBeGreaterThan(0);
  });

  it('should classify a business prompt as BUSINESS', () => {
    const result = classifier.classify('Create a business strategy for our quarterly financial report');
    expect(result.broadCategory).toBe('BUSINESS');
  });

  it('should classify an educational prompt as EDUCATIONAL', () => {
    const result = classifier.classify('Explain the concept of photosynthesis for a student assignment');
    expect(result.broadCategory).toBe('EDUCATIONAL');
    expect(result.benchmarkScores['MMLU']).toBeGreaterThan(0);
  });

  it('should classify a conversational prompt as CONVERSATIONAL', () => {
    const result = classifier.classify('What should I recommend to my friend for their decision?');
    expect(result.broadCategory).toBe('CONVERSATIONAL');
  });

  it('should always include base scores for general benchmarks', () => {
    const result = classifier.classify('Write code to deploy a docker container');
    // Even for technical prompts, general benchmarks should have base scores
    expect(result.benchmarkScores['MMLU']).toBeDefined();
    expect(result.benchmarkScores['Chatbot Arena (LMSys)']).toBeDefined();
    expect(result.benchmarkScores['HellaSwag']).toBeDefined();
  });

  it('should have confidence between 0 and 1', () => {
    const result = classifier.classify('Help me write a Python function for machine learning classification');
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
  });

  it('should handle empty prompt gracefully', () => {
    const result = classifier.classify('');
    expect(result.broadCategory).toBeTruthy();
    expect(result.classifierUsed).toBe('keyword');
  });

  it('should handle math-related prompts', () => {
    const result = classifier.classify('Calculate the formula for the probability equation');
    expect(result.broadCategory).toBe('TECHNICAL');
    expect(result.subcategory).toBe('MATHEMATICAL_SCIENTIFIC');
    expect(result.benchmarkScores['GSM8K']).toBeGreaterThan(0);
  });
});
