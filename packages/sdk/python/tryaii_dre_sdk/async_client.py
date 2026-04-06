"""
AsyncDREClient -- async version of DREClient for TryAii-DRE.

Provides the same interface as DREClient but with async/await support.
Uses asyncio.to_thread() for CPU-bound Router calls and httpx.AsyncClient
for non-blocking API calls.

Usage:
    from tryaii_dre_sdk.async_client import AsyncDREClient

    client = AsyncDREClient(api_key="sk-or-...")
    response = await client.chat("Write a quicksort in Python")
    print(response.model_used, response.content)
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import AsyncGenerator, Optional

import httpx

from tryaii_dre import Router, RouteResult
from tryaii_dre.config import TryaiiDreConfig
from tryaii_dre.integrations.openrouter import (
    MODEL_ID_TO_OPENROUTER,
    OpenRouterResponse,
)
from tryaii_dre.scoring.priorities import Priorities


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class AsyncDREClient:
    """
    Async high-level client that combines routing and API calls.

    Same interface as DREClient but all methods are async. Router calls
    are offloaded to a thread pool via asyncio.to_thread(), and API calls
    use httpx.AsyncClient for true async I/O.

    Args:
        api_key: OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.
        priorities: Default priorities for all routing calls.
        embedding_model: Sentence-transformers model name for embeddings.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        priorities: Optional[Priorities] = None,
        embedding_model: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._default_priorities = priorities
        self._embedding_model = embedding_model

        # Build config
        config = TryaiiDreConfig()
        if embedding_model:
            config = TryaiiDreConfig(embedding_model=embedding_model)

        # Core router (sync -- will be called via asyncio.to_thread)
        self._router = Router(config=config)

        # Async HTTP client (lazy-initialized)
        self._http_client: Optional[httpx.AsyncClient] = None

    def _ensure_http_client(self) -> httpx.AsyncClient:
        """Lazy-initialize the async HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=OPENROUTER_BASE_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "X-Title": "tryaii-dre-sdk",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._http_client

    @staticmethod
    def _resolve_model(model_id: str) -> str:
        """Convert TryAii-DRE model ID to OpenRouter slug."""
        return MODEL_ID_TO_OPENROUTER.get(model_id, model_id)

    @property
    def router(self) -> Router:
        """Access the underlying Router instance."""
        return self._router

    async def route(
        self,
        prompt: str,
        priorities: Optional[Priorities] = None,
        top_k: int = 5,
    ) -> RouteResult:
        """
        Route a prompt without making an API call (async).

        The Router's classify/score logic is CPU-bound, so it is
        offloaded to a thread to avoid blocking the event loop.

        Args:
            prompt: The user message to classify and route.
            priorities: Override default priorities for this call.
            top_k: Number of top models to include in results.

        Returns:
            RouteResult with best_model, scores, and classification.
        """
        prio = priorities or self._default_priorities
        return await asyncio.to_thread(
            self._router.route, prompt, priorities=prio, top_k=top_k
        )

    async def chat(
        self,
        prompt: str,
        priorities: Optional[Priorities] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> OpenRouterResponse:
        """
        Route the prompt to the best model and return the AI response (async).

        Args:
            prompt: The user message to route and send.
            priorities: Override default priorities for this call.
            system_message: Optional system prompt.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in the response.

        Returns:
            OpenRouterResponse with content, model_used, and routing info.
        """
        # Route in a thread (CPU-bound)
        route_result = await self.route(prompt, priorities=priorities)
        model_id = route_result.best_model
        reasoning = route_result.scores[0].reasoning if route_result.scores else ""
        openrouter_model = self._resolve_model(model_id)

        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": openrouter_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Async API call
        client = self._ensure_http_client()
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        return OpenRouterResponse(
            content=content,
            model_used=model_id,
            openrouter_model=openrouter_model,
            route_reasoning=reasoning,
            usage=usage,
            raw_response=data,
        )

    async def stream(
        self,
        prompt: str,
        priorities: Optional[Priorities] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Route the prompt to the best model and stream the response (async).

        Yields content chunks as they arrive from the API.

        Args:
            prompt: The user message to route and send.
            priorities: Override default priorities for this call.
            system_message: Optional system prompt.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in the response.

        Yields:
            String chunks of the response content.
        """
        # Route in a thread (CPU-bound)
        route_result = await self.route(prompt, priorities=priorities)
        model_id = route_result.best_model
        openrouter_model = self._resolve_model(model_id)

        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": openrouter_model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Async streaming
        client = self._ensure_http_client()
        async with client.stream("POST", "/chat/completions", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    async def route_and_chat(
        self,
        prompt: str,
        priorities: Optional[Priorities] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> tuple[RouteResult, OpenRouterResponse]:
        """
        Route a prompt and get the AI response, returning both (async).

        Args:
            prompt: The user message to route and send.
            priorities: Override default priorities for this call.
            system_message: Optional system prompt.
            temperature: Sampling temperature (0.0 to 2.0).
            max_tokens: Maximum tokens in the response.

        Returns:
            Tuple of (RouteResult, OpenRouterResponse).
        """
        # Route first
        route_result = await self.route(prompt, priorities=priorities)
        model_id = route_result.best_model
        reasoning = route_result.scores[0].reasoning if route_result.scores else ""
        openrouter_model = self._resolve_model(model_id)

        # Build messages
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": openrouter_model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Async API call
        client = self._ensure_http_client()
        response = await client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        api_response = OpenRouterResponse(
            content=content,
            model_used=model_id,
            openrouter_model=openrouter_model,
            route_reasoning=reasoning,
            usage=usage,
            raw_response=data,
        )

        return route_result, api_response

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self) -> AsyncDREClient:
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
