"""
TryAii-DRE SDK -- High-level client for AI model routing.

Provides a unified DREClient that wraps the core Router and OpenRouter
integration into a single interface with async support and middleware.

Usage:
    from tryaii_dre_sdk import DREClient

    client = DREClient(api_key="sk-or-...")
    response = client.chat("Write a Python quicksort")
    print(response.content)
"""

from tryaii_dre_sdk.client import DREClient
from tryaii_dre_sdk.async_client import AsyncDREClient

__version__ = "0.1.0"

__all__ = [
    "DREClient",
    "AsyncDREClient",
    "__version__",
]
