"""Tests for the keyword classifier."""

from tryaii_dre.classifiers.keyword import KeywordClassifier


class TestKeywordClassifier:
    def setup_method(self):
        self.classifier = KeywordClassifier()

    def test_is_always_ready(self):
        assert self.classifier.is_ready()

    def test_classify_code_prompt(self):
        result = self.classifier.classify("Write a Python function to sort an array")
        assert result.broad_category == "TECHNICAL"
        assert result.confidence > 0
        assert "HumanEval" in result.benchmark_scores or "SWE-bench" in result.benchmark_scores

    def test_classify_creative_prompt(self):
        result = self.classifier.classify("Write a poem about the ocean at sunset")
        assert result.broad_category == "CREATIVE"
        assert result.confidence > 0

    def test_classify_business_prompt(self):
        result = self.classifier.classify("Create a financial budget report for Q3")
        assert result.broad_category == "BUSINESS"

    def test_classify_educational_prompt(self):
        result = self.classifier.classify("Explain how photosynthesis works in plants")
        assert result.broad_category == "EDUCATIONAL"

    def test_classify_conversational_prompt(self):
        result = self.classifier.classify("What do you think I should do about this problem?")
        assert result.broad_category == "CONVERSATIONAL"

    def test_returns_benchmark_scores(self):
        result = self.classifier.classify("Write a recursive fibonacci function")
        assert len(result.benchmark_scores) > 0
        # Should always include baseline benchmarks
        assert "MMLU" in result.benchmark_scores or "Chatbot Arena (LMSys)" in result.benchmark_scores

    def test_classifier_used_is_keyword(self):
        result = self.classifier.classify("Hello world")
        assert result.classifier_used == "keyword"

    def test_processing_time_recorded(self):
        result = self.classifier.classify("test prompt")
        assert result.processing_time_ms >= 0

    def test_subcategory_code_technical(self):
        result = self.classifier.classify(
            "Debug this API function and fix the class method"
        )
        assert result.broad_category == "TECHNICAL"
        assert result.subcategory == "CODE_TECHNICAL"

    def test_subcategory_math(self):
        result = self.classifier.classify(
            "Calculate the probability using this formula and equation"
        )
        assert result.subcategory == "MATHEMATICAL_SCIENTIFIC"

    def test_empty_prompt(self):
        result = self.classifier.classify("")
        assert result.confidence == 0 or result.broad_category != ""

    def test_top_benchmarks_sorted(self):
        result = self.classifier.classify("Write code to implement a binary tree")
        top = result.top_benchmarks
        # Should be sorted descending by score
        scores = [s for _, s in top]
        assert scores == sorted(scores, reverse=True)
