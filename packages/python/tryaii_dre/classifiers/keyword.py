"""
Keyword-based hierarchical classifier.

Uses pattern matching with confidence scoring as a fast fallback
when embedding classification is unavailable or low-confidence.
No external dependencies -- works offline with zero setup.
"""

from __future__ import annotations

import math
import re
import time

from tryaii_dre.classifiers.base import BaseClassifier, ClassificationResult

# Broad category -> keyword patterns
BROAD_PATTERNS: dict[str, list[str]] = {
    "TECHNICAL": [
        "code", "programming", "algorithm", "debug", "api", "database", "sql",
        "python", "javascript", "react", "node", "function", "class", "method",
        "variable", "loop", "array", "object", "math", "calculate", "formula",
        "equation", "statistics", "data analysis", "machine learning", "ai",
        "neural network", "regression", "classification", "clustering",
        "dataset", "deploy", "docker", "kubernetes", "terraform", "ci/cd",
    ],
    "CREATIVE": [
        "story", "write", "creative", "poem", "poetry", "narrative", "character",
        "plot", "dialogue", "design", "color", "layout", "visual", "aesthetic",
        "brand", "logo", "artwork", "graphic", "brainstorm", "idea", "innovate",
        "concept", "inspiration", "imagine", "invent", "original", "music",
        "song", "melody", "rhythm", "video", "film", "script", "performance", "art",
    ],
    "BUSINESS": [
        "business", "strategy", "plan", "market", "revenue", "profit", "growth",
        "competition", "budget", "financial", "investment", "roi", "cost",
        "expense", "accounting", "finance", "legal", "contract", "compliance",
        "regulation", "policy", "law", "terms", "agreement", "email",
        "presentation", "meeting", "proposal", "report", "communication",
        "professional", "management", "team", "leadership", "operations",
        "process", "workflow", "organization", "summarize", "summary",
        "memo", "stakeholder", "client", "invoice", "quarterly",
    ],
    "EDUCATIONAL": [
        "explain", "teach", "learn", "education", "instruction", "lesson",
        "concept", "understand", "homework", "assignment", "study", "exam",
        "test", "quiz", "grade", "student", "school", "research", "paper",
        "thesis", "academic", "scholar", "methodology", "analysis", "review",
    ],
    "CONVERSATIONAL": [
        "advice", "help", "recommend", "suggest", "opinion", "think", "feel",
        "personal", "life", "relationship", "friend", "family", "decision",
        "choice", "problem", "solution", "guidance", "best", "better",
        "should", "prefer",
    ],
}

# Subcategory patterns nested under broad categories
SUBCATEGORY_PATTERNS: dict[str, dict[str, list[str]]] = {
    "TECHNICAL": {
        "CODE_TECHNICAL": [
            "code", "programming", "debug", "api", "function", "class", "method",
            "variable", "algorithm", "data structure", "framework", "library",
            "repository", "git", "deploy",
        ],
        "MATHEMATICAL_SCIENTIFIC": [
            "math", "calculate", "formula", "equation", "statistics", "probability",
            "theorem", "physics", "chemistry", "biology", "science", "research",
            "hypothesis", "experiment",
        ],
        "DATA_SCIENCE": [
            "data", "dataset", "analysis", "machine learning", "ai", "model",
            "neural network", "regression", "classification", "clustering",
            "visualization", "pandas", "numpy",
        ],
    },
    "CREATIVE": {
        "WRITING_LITERARY": [
            "story", "write", "novel", "poem", "poetry", "narrative", "character",
            "plot", "dialogue", "fiction", "non-fiction", "essay", "article",
            "blog", "content",
        ],
        "VISUAL_DESIGN": [
            "design", "color", "layout", "visual", "aesthetic", "brand", "logo",
            "ui", "ux", "graphic", "typography", "illustration", "image", "photo",
            "artwork",
        ],
        "CREATIVE_IDEATION": [
            "brainstorm", "idea", "innovate", "concept", "inspiration", "imagine",
            "invent", "original", "creative thinking", "alternative",
            "possibility",
        ],
        "MEDIA_ARTS": [
            "music", "song", "melody", "rhythm", "video", "film", "script",
            "performance", "theater", "dance", "photography", "animation",
            "multimedia",
        ],
    },
    "BUSINESS": {
        "STRATEGY_PLANNING": [
            "strategy", "plan", "business plan", "roadmap", "goal", "objective",
            "vision", "mission", "competitive", "market analysis", "swot",
            "growth", "expansion",
        ],
        "FINANCIAL_ANALYSIS": [
            "financial", "budget", "investment", "roi", "revenue", "profit",
            "cost", "expense", "cash flow", "forecast", "valuation", "accounting",
            "finance", "pricing",
        ],
        "LEGAL_COMPLIANCE": [
            "legal", "law", "contract", "agreement", "compliance", "regulation",
            "policy", "terms", "conditions", "intellectual property", "patent",
            "copyright", "gdpr",
        ],
        "PROFESSIONAL_COMMUNICATION": [
            "email", "presentation", "meeting", "proposal", "report", "memo",
            "letter", "communication", "professional", "corporate", "client",
            "stakeholder", "summarize", "summary", "brief", "digest",
        ],
    },
    "EDUCATIONAL": {
        "ACADEMIC_INSTRUCTION": [
            "explain", "teach", "lesson", "instruction", "lecture", "tutorial",
            "concept", "theory", "understand",
        ],
        "STUDY_ASSISTANCE": [
            "homework", "assignment", "study", "exam", "test", "quiz", "grade",
            "student", "school",
        ],
        "RESEARCH_METHODOLOGY": [
            "research", "paper", "thesis", "academic", "scholar", "methodology",
            "analysis", "review", "peer review", "citation",
        ],
    },
    "CONVERSATIONAL": {
        "PERSONAL_ADVICE": [
            "advice", "personal", "life", "relationship", "friend", "family",
            "decision", "guidance", "feel", "emotion",
        ],
        "RECOMMENDATIONS": [
            "recommend", "suggest", "best", "better", "compare", "which",
            "should i", "top", "review", "rating",
        ],
    },
}

# Map categories -> benchmark names for score output
CATEGORY_TO_BENCHMARKS: dict[str, list[str]] = {
    "CODE_TECHNICAL": ["HumanEval", "SWE-bench", "LiveBench"],
    "MATHEMATICAL_SCIENTIFIC": ["GSM8K", "DROP", "ARC"],
    "DATA_SCIENCE": ["HumanEval", "SWE-bench", "DROP"],
    "WRITING_LITERARY": ["MT-Bench", "Chatbot Arena (LMSys)"],
    "VISUAL_DESIGN": ["MT-Bench"],
    "CREATIVE_IDEATION": ["MT-Bench", "Chatbot Arena (LMSys)"],
    "MEDIA_ARTS": ["MT-Bench"],
    "STRATEGY_PLANNING": ["MMLU", "SuperGLUE"],
    "FINANCIAL_ANALYSIS": ["GSM8K", "DROP", "MMLU"],
    "LEGAL_COMPLIANCE": ["MMLU", "TruthfulQA"],
    "PROFESSIONAL_COMMUNICATION": ["SuperGLUE", "MT-Bench"],
    "ACADEMIC_INSTRUCTION": ["MMLU", "ARC"],
    "STUDY_ASSISTANCE": ["MMLU", "GSM8K"],
    "RESEARCH_METHODOLOGY": ["MMLU", "TruthfulQA"],
    "PERSONAL_ADVICE": ["Chatbot Arena (LMSys)", "TruthfulQA", "HellaSwag"],
    "RECOMMENDATIONS": ["Chatbot Arena (LMSys)", "HellaSwag"],
    # Broad-level fallbacks
    "TECHNICAL": ["HumanEval", "SWE-bench", "GSM8K", "LiveBench"],
    "CREATIVE": ["MT-Bench", "Chatbot Arena (LMSys)"],
    "BUSINESS": ["MMLU", "SuperGLUE", "DROP"],
    "EDUCATIONAL": ["MMLU", "ARC", "GSM8K"],
    "CONVERSATIONAL": ["Chatbot Arena (LMSys)", "HellaSwag", "TruthfulQA"],
}


def _score_category(text_lower: str, keywords: list[str]) -> float:
    """Calculate keyword match score for a category."""
    if not keywords:
        return 0.0

    matches = 0
    weighted_matches = 0.0

    for kw in keywords:
        # Use word-boundary matching to avoid substring false positives
        pattern = r'\b' + re.escape(kw) + r'\b'
        if re.search(pattern, text_lower):
            matches += 1
            # Longer keywords are more specific -> higher weight
            weight = 1.5 if len(kw) > 5 else 1.0
            weighted_matches += weight

    if matches == 0:
        return 0.0

    # Square-root normalization + match bonus
    base_score = math.sqrt(weighted_matches / len(keywords))

    # Bonus for multiple matches
    if matches >= 3:
        base_score += 0.2
    elif matches >= 2:
        base_score += 0.1

    return min(1.0, base_score)


class KeywordClassifier(BaseClassifier):
    """
    Fast keyword-based classifier using hierarchical pattern matching.

    Two-stage classification:
        1. Broad category (TECHNICAL, CREATIVE, BUSINESS, EDUCATIONAL, CONVERSATIONAL)
        2. Subcategory refinement (CODE_TECHNICAL, WRITING_LITERARY, etc.)

    Output is mapped to benchmark similarity scores for compatibility
    with the ScoringEngine.
    """

    def classify(self, prompt: str) -> ClassificationResult:
        start = time.time()
        text_lower = prompt.lower()

        # Stage 1: Broad category
        broad_scores: dict[str, float] = {}
        for category, keywords in BROAD_PATTERNS.items():
            broad_scores[category] = _score_category(text_lower, keywords)

        best_broad = max(broad_scores, key=broad_scores.get)  # type: ignore[arg-type]
        broad_confidence = broad_scores[best_broad]

        # Stage 2: Subcategory (if broad confidence is meaningful)
        best_sub = ""
        sub_confidence = 0.0

        if broad_confidence > 0.1 and best_broad in SUBCATEGORY_PATTERNS:
            sub_scores: dict[str, float] = {}
            for sub, keywords in SUBCATEGORY_PATTERNS[best_broad].items():
                sub_scores[sub] = _score_category(text_lower, keywords)

            if sub_scores:
                best_sub = max(sub_scores, key=sub_scores.get)  # type: ignore[arg-type]
                sub_confidence = sub_scores[best_sub]

        # Convert to benchmark scores
        final_category = best_sub if best_sub and sub_confidence > 0.05 else best_broad
        benchmark_names = CATEGORY_TO_BENCHMARKS.get(final_category, [])

        benchmark_scores: dict[str, float] = {}
        confidence = sub_confidence if sub_confidence > 0.05 else broad_confidence

        for bench in benchmark_names:
            benchmark_scores[bench] = confidence

        # Add a base score for general benchmarks
        for bench in ["MMLU", "Chatbot Arena (LMSys)", "HellaSwag"]:
            if bench not in benchmark_scores:
                benchmark_scores[bench] = max(0.05, broad_confidence * 0.3)

        elapsed_ms = (time.time() - start) * 1000

        return ClassificationResult(
            benchmark_scores=benchmark_scores,
            broad_category=best_broad,
            subcategory=best_sub,
            confidence=confidence,
            classifier_used="keyword",
            cache_hit=False,
            processing_time_ms=elapsed_ms,
        )

    def is_ready(self) -> bool:
        return True  # Always ready -- no initialization needed
