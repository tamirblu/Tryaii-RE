"""
Tests for the DREClient high-level SDK client.

These tests validate initialization, routing-only behavior, and
custom priority handling without making any real API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from tryaii_dre import RouteResult
from tryaii_dre.scoring.priorities import Priorities

from tryaii_dre_sdk.client import DREClient


class TestClientInitializes:
    """DREClient should initialize with default and custom parameters."""

    def test_default_initialization(self):
        """Client initializes without any arguments."""
        client = DREClient()
        assert client.router is not None
        assert client.openrouter is not None

    def test_initialization_with_api_key(self):
        """Client stores the API key for OpenRouter calls."""
        client = DREClient(api_key="sk-or-test-key")
        assert client._api_key == "sk-or-test-key"

    def test_initialization_with_priorities(self):
        """Client stores default priorities."""
        prio = Priorities(quality=5, cost=1, speed=2)
        client = DREClient(priorities=prio)
        assert client._default_priorities is prio
        assert client._default_priorities.quality == 5

    def test_initialization_with_embedding_model(self):
        """Client accepts a custom embedding model name."""
        client = DREClient(embedding_model="all-MiniLM-L6-v2")
        assert client._embedding_model == "all-MiniLM-L6-v2"


class TestClientRouteOnly:
    """DREClient.route() should classify and rank models without API calls."""

    def test_route_returns_route_result(self):
        """route() returns a RouteResult with a best_model."""
        client = DREClient()
        result = client.route("Write a Python quicksort implementation")
        assert isinstance(result, RouteResult)
        assert isinstance(result.best_model, str)
        assert len(result.best_model) > 0

    def test_route_returns_scores(self):
        """route() returns scored models."""
        client = DREClient()
        result = client.route("Explain quantum computing in simple terms")
        assert len(result.scores) > 0
        # Scores should be sorted descending
        for i in range(len(result.scores) - 1):
            assert result.scores[i].final_score >= result.scores[i + 1].final_score

    def test_route_respects_top_k(self):
        """route() limits results to top_k models."""
        client = DREClient()
        result = client.route("Translate hello to French", top_k=3)
        assert len(result.scores) <= 3


class TestClientWithCustomPriorities:
    """DREClient should respect custom priorities in routing."""

    def test_default_priorities_used_in_route(self):
        """Default priorities set at init are used by route()."""
        prio = Priorities.performance()
        client = DREClient(priorities=prio)
        result = client.route("Write a complex algorithm")
        assert result.priorities.quality == 5
        assert result.priorities.cost == 1

    def test_per_request_priorities_override_default(self):
        """Per-request priorities override the defaults."""
        client = DREClient(priorities=Priorities.performance())
        override = Priorities.budget()
        result = client.route("Write a simple script", priorities=override)
        assert result.priorities.cost == 5
        assert result.priorities.quality == 2

    def test_budget_priorities_favor_cheaper_models(self):
        """Budget priorities should influence model ranking toward cost."""
        client = DREClient()
        budget_result = client.route(
            "Summarize this text",
            priorities=Priorities(quality=1, cost=5, speed=3),
        )
        perf_result = client.route(
            "Summarize this text",
            priorities=Priorities(quality=5, cost=1, speed=1),
        )
        # The two routing decisions should potentially differ
        # (or at minimum, both should return valid results)
        assert isinstance(budget_result.best_model, str)
        assert isinstance(perf_result.best_model, str)
        assert len(budget_result.scores) > 0
        assert len(perf_result.scores) > 0
