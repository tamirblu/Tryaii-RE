/**
 * Keyword-based hierarchical classifier.
 *
 * Uses pattern matching with confidence scoring as a fast fallback
 * when embedding classification is unavailable or low-confidence.
 * No external dependencies -- works offline with zero setup.
 */

import { BaseClassifier, ClassificationResult } from './base.js';

/** Broad category -> keyword patterns. */
export const BROAD_PATTERNS: Record<string, string[]> = {
  TECHNICAL: [
    'code', 'programming', 'algorithm', 'debug', 'api', 'database', 'sql',
    'python', 'javascript', 'react', 'node', 'function', 'class', 'method',
    'variable', 'loop', 'array', 'object', 'math', 'calculate', 'formula',
    'equation', 'statistics', 'data analysis', 'machine learning', 'ai',
    'model', 'neural network', 'regression', 'classification', 'clustering',
    'dataset', 'deploy', 'docker', 'kubernetes', 'terraform', 'ci/cd',
  ],
  CREATIVE: [
    'story', 'write', 'creative', 'poem', 'poetry', 'narrative', 'character',
    'plot', 'dialogue', 'design', 'color', 'layout', 'visual', 'aesthetic',
    'brand', 'logo', 'artwork', 'graphic', 'brainstorm', 'idea', 'innovate',
    'concept', 'inspiration', 'imagine', 'invent', 'original', 'music',
    'song', 'melody', 'rhythm', 'video', 'film', 'script', 'performance', 'art',
  ],
  BUSINESS: [
    'business', 'strategy', 'plan', 'market', 'revenue', 'profit', 'growth',
    'competition', 'budget', 'financial', 'investment', 'roi', 'cost',
    'expense', 'accounting', 'finance', 'legal', 'contract', 'compliance',
    'regulation', 'policy', 'law', 'terms', 'agreement', 'email',
    'presentation', 'meeting', 'proposal', 'report', 'communication',
    'professional', 'management', 'team', 'leadership', 'operations',
    'process', 'workflow', 'organization', 'summarize', 'summary',
    'memo', 'stakeholder', 'client', 'invoice', 'quarterly',
  ],
  EDUCATIONAL: [
    'explain', 'teach', 'learn', 'education', 'instruction', 'lesson',
    'concept', 'understand', 'homework', 'assignment', 'study', 'exam',
    'test', 'quiz', 'grade', 'student', 'school', 'research', 'paper',
    'thesis', 'academic', 'scholar', 'methodology', 'analysis', 'review',
  ],
  CONVERSATIONAL: [
    'advice', 'help', 'recommend', 'suggest', 'opinion', 'think', 'feel',
    'personal', 'life', 'relationship', 'friend', 'family', 'decision',
    'choice', 'problem', 'solution', 'guidance', 'what', 'how', 'why',
    'when', 'where', 'tell me', 'can you', 'please', 'would you',
    'could you', 'best', 'better', 'good', 'bad', 'should', 'need',
    'want', 'like', 'love', 'hate', 'prefer',
  ],
};

/** Subcategory patterns nested under broad categories. */
export const SUBCATEGORY_PATTERNS: Record<string, Record<string, string[]>> = {
  TECHNICAL: {
    CODE_TECHNICAL: [
      'code', 'programming', 'debug', 'api', 'function', 'class', 'method',
      'variable', 'algorithm', 'data structure', 'framework', 'library',
      'repository', 'git', 'deploy',
    ],
    MATHEMATICAL_SCIENTIFIC: [
      'math', 'calculate', 'formula', 'equation', 'statistics', 'probability',
      'theorem', 'physics', 'chemistry', 'biology', 'science', 'research',
      'hypothesis', 'experiment',
    ],
    DATA_SCIENCE: [
      'data', 'dataset', 'analysis', 'machine learning', 'ai', 'model',
      'neural network', 'regression', 'classification', 'clustering',
      'visualization', 'pandas', 'numpy',
    ],
  },
  CREATIVE: {
    WRITING_LITERARY: [
      'story', 'write', 'novel', 'poem', 'poetry', 'narrative', 'character',
      'plot', 'dialogue', 'fiction', 'non-fiction', 'essay', 'article',
      'blog', 'content',
    ],
    VISUAL_DESIGN: [
      'design', 'color', 'layout', 'visual', 'aesthetic', 'brand', 'logo',
      'ui', 'ux', 'graphic', 'typography', 'illustration', 'image', 'photo',
      'artwork',
    ],
    CREATIVE_IDEATION: [
      'brainstorm', 'idea', 'innovate', 'concept', 'inspiration', 'imagine',
      'invent', 'original', 'creative thinking', 'solution', 'alternative',
      'possibility',
    ],
    MEDIA_ARTS: [
      'music', 'song', 'melody', 'rhythm', 'video', 'film', 'script',
      'performance', 'theater', 'dance', 'photography', 'animation',
      'multimedia',
    ],
  },
  BUSINESS: {
    STRATEGY_PLANNING: [
      'strategy', 'plan', 'business plan', 'roadmap', 'goal', 'objective',
      'vision', 'mission', 'competitive', 'market analysis', 'swot',
      'growth', 'expansion',
    ],
    FINANCIAL_ANALYSIS: [
      'financial', 'budget', 'investment', 'roi', 'revenue', 'profit',
      'cost', 'expense', 'cash flow', 'forecast', 'valuation', 'accounting',
      'finance', 'pricing',
    ],
    LEGAL_COMPLIANCE: [
      'legal', 'law', 'contract', 'agreement', 'compliance', 'regulation',
      'policy', 'terms', 'conditions', 'intellectual property', 'patent',
      'copyright', 'gdpr',
    ],
    PROFESSIONAL_COMMUNICATION: [
      'email', 'presentation', 'meeting', 'proposal', 'report', 'memo',
      'letter', 'communication', 'professional', 'corporate', 'client',
      'stakeholder', 'summarize', 'summary', 'brief', 'digest',
    ],
  },
  EDUCATIONAL: {
    ACADEMIC_INSTRUCTION: [
      'explain', 'teach', 'lesson', 'instruction', 'lecture', 'tutorial',
      'concept', 'theory', 'understand',
    ],
    STUDY_ASSISTANCE: [
      'homework', 'assignment', 'study', 'exam', 'test', 'quiz', 'grade',
      'student', 'school',
    ],
    RESEARCH_METHODOLOGY: [
      'research', 'paper', 'thesis', 'academic', 'scholar', 'methodology',
      'analysis', 'review', 'peer review', 'citation',
    ],
  },
  CONVERSATIONAL: {
    PERSONAL_ADVICE: [
      'advice', 'personal', 'life', 'relationship', 'friend', 'family',
      'decision', 'guidance', 'feel', 'emotion',
    ],
    RECOMMENDATIONS: [
      'recommend', 'suggest', 'best', 'better', 'compare', 'which',
      'should i', 'top', 'review', 'rating',
    ],
  },
};

/** Map categories -> benchmark names for score output. */
export const CATEGORY_TO_BENCHMARKS: Record<string, string[]> = {
  CODE_TECHNICAL: ['HumanEval', 'SWE-bench'],
  MATHEMATICAL_SCIENTIFIC: ['GSM8K', 'DROP', 'ARC'],
  DATA_SCIENCE: ['HumanEval', 'SWE-bench', 'DROP'],
  WRITING_LITERARY: ['MT-Bench', 'Chatbot Arena (LMSys)'],
  VISUAL_DESIGN: ['MT-Bench'],
  CREATIVE_IDEATION: ['MT-Bench', 'Chatbot Arena (LMSys)'],
  MEDIA_ARTS: ['MT-Bench'],
  STRATEGY_PLANNING: ['MMLU', 'SuperGLUE'],
  FINANCIAL_ANALYSIS: ['GSM8K', 'DROP', 'MMLU'],
  LEGAL_COMPLIANCE: ['MMLU', 'TruthfulQA'],
  PROFESSIONAL_COMMUNICATION: ['SuperGLUE', 'MT-Bench'],
  ACADEMIC_INSTRUCTION: ['MMLU', 'ARC'],
  STUDY_ASSISTANCE: ['MMLU', 'GSM8K'],
  RESEARCH_METHODOLOGY: ['MMLU', 'TruthfulQA'],
  PERSONAL_ADVICE: ['Chatbot Arena (LMSys)', 'TruthfulQA', 'HellaSwag'],
  RECOMMENDATIONS: ['Chatbot Arena (LMSys)', 'HellaSwag'],
  // Broad-level fallbacks
  TECHNICAL: ['HumanEval', 'SWE-bench', 'GSM8K'],
  CREATIVE: ['MT-Bench', 'Chatbot Arena (LMSys)'],
  BUSINESS: ['MMLU', 'SuperGLUE', 'DROP'],
  EDUCATIONAL: ['MMLU', 'ARC', 'GSM8K'],
  CONVERSATIONAL: ['Chatbot Arena (LMSys)', 'HellaSwag', 'TruthfulQA'],
};

/** Calculate keyword match score for a category. */
function scoreCategory(textLower: string, keywords: string[]): number {
  if (keywords.length === 0) return 0.0;

  let matches = 0;
  let weightedMatches = 0.0;

  for (const kw of keywords) {
    if (textLower.includes(kw)) {
      matches += 1;
      // Longer keywords are more specific -> higher weight
      const weight = kw.length > 5 ? 1.5 : 1.0;
      weightedMatches += weight;
    }
  }

  if (matches === 0) return 0.0;

  // Square-root normalization + match bonus
  let baseScore = Math.sqrt(weightedMatches / keywords.length);

  // Bonus for multiple matches
  if (matches >= 3) {
    baseScore += 0.2;
  } else if (matches >= 2) {
    baseScore += 0.1;
  }

  return Math.min(1.0, baseScore);
}

/**
 * Fast keyword-based classifier using hierarchical pattern matching.
 *
 * Two-stage classification:
 *   1. Broad category (TECHNICAL, CREATIVE, BUSINESS, EDUCATIONAL, CONVERSATIONAL)
 *   2. Subcategory refinement (CODE_TECHNICAL, WRITING_LITERARY, etc.)
 *
 * Output is mapped to benchmark similarity scores for compatibility
 * with the ScoringEngine.
 */
export class KeywordClassifier extends BaseClassifier {
  classify(prompt: string): ClassificationResult {
    const start = performance.now();
    const textLower = prompt.toLowerCase();

    // Stage 1: Broad category
    const broadScores: Record<string, number> = {};
    for (const [category, keywords] of Object.entries(BROAD_PATTERNS)) {
      broadScores[category] = scoreCategory(textLower, keywords);
    }

    let bestBroad = '';
    let broadConfidence = 0;
    for (const [cat, score] of Object.entries(broadScores)) {
      if (score > broadConfidence) {
        broadConfidence = score;
        bestBroad = cat;
      }
    }
    if (!bestBroad) bestBroad = 'CONVERSATIONAL';

    // Stage 2: Subcategory (if broad confidence is meaningful)
    let bestSub = '';
    let subConfidence = 0.0;

    if (broadConfidence > 0.1 && SUBCATEGORY_PATTERNS[bestBroad]) {
      const subScores: Record<string, number> = {};
      for (const [sub, keywords] of Object.entries(SUBCATEGORY_PATTERNS[bestBroad])) {
        subScores[sub] = scoreCategory(textLower, keywords);
      }

      for (const [sub, score] of Object.entries(subScores)) {
        if (score > subConfidence) {
          subConfidence = score;
          bestSub = sub;
        }
      }
    }

    // Convert to benchmark scores
    const finalCategory = bestSub && subConfidence > 0.05 ? bestSub : bestBroad;
    const benchmarkNames = CATEGORY_TO_BENCHMARKS[finalCategory] ?? [];

    const benchmarkScores: Record<string, number> = {};
    const confidence = subConfidence > 0.05 ? subConfidence : broadConfidence;

    for (const bench of benchmarkNames) {
      benchmarkScores[bench] = confidence;
    }

    // Add a base score for general benchmarks
    for (const bench of ['MMLU', 'Chatbot Arena (LMSys)', 'HellaSwag']) {
      if (!(bench in benchmarkScores)) {
        benchmarkScores[bench] = Math.max(0.05, broadConfidence * 0.3);
      }
    }

    const elapsedMs = performance.now() - start;

    return {
      benchmarkScores,
      broadCategory: bestBroad,
      subcategory: bestSub,
      confidence,
      classifierUsed: 'keyword',
      cacheHit: false,
      processingTimeMs: elapsedMs,
    };
  }

  isReady(): boolean {
    return true; // Always ready -- no initialization needed
  }
}
