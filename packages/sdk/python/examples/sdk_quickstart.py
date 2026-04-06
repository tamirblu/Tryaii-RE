"""
TryAii-DRE SDK -- Quick Start Example

Demonstrates how to use the DREClient for routing and chat in just
a few lines of code. Set your OPENROUTER_API_KEY environment variable
before running.

Usage:
    export OPENROUTER_API_KEY="sk-or-..."
    python sdk_quickstart.py
"""

from tryaii_dre import Priorities
from tryaii_dre_sdk import DREClient


def main():
    # -- Basic routing (no API call) --
    client = DREClient()

    print("=== Route Only (no API call) ===")
    result = client.route("Write a Python function to merge two sorted arrays")
    print(f"Best model: {result.best_model}")
    print(f"Score:      {result.best_score:.4f}")
    print(f"Reasoning:  {result.best_reasoning}")
    print(f"Top 3:      {result.top_k[:3]}")
    print()

    # -- Route with custom priorities --
    print("=== Budget-Focused Routing ===")
    budget_result = client.route(
        "Summarize this article about climate change",
        priorities=Priorities.budget(),
    )
    print(f"Best model (budget): {budget_result.best_model}")
    print()

    print("=== Performance-Focused Routing ===")
    perf_result = client.route(
        "Summarize this article about climate change",
        priorities=Priorities.performance(),
    )
    print(f"Best model (performance): {perf_result.best_model}")
    print()

    # -- Chat (requires OPENROUTER_API_KEY) --
    # Uncomment the following to make actual API calls:
    #
    # client = DREClient(api_key="sk-or-...")
    #
    # print("=== Chat ===")
    # response = client.chat("Write a haiku about programming")
    # print(f"Model: {response.model_used}")
    # print(f"Response: {response.content}")
    # print()
    #
    # print("=== Streaming ===")
    # for chunk in client.stream("Explain recursion in 3 sentences"):
    #     print(chunk, end="", flush=True)
    # print()
    #
    # print("=== Route and Chat ===")
    # route_result, response = client.route_and_chat(
    #     "What is the capital of France?",
    #     system_message="You are a helpful geography assistant.",
    # )
    # print(f"Routed to: {route_result.best_model} (score: {route_result.best_score:.4f})")
    # print(f"Response: {response.content}")


if __name__ == "__main__":
    main()
