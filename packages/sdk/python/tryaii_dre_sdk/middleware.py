"""
ASGI middleware for TryAii-DRE.

Adds routing headers (X-DRE-Model, X-DRE-Score) to HTTP responses
in FastAPI, Starlette, or any ASGI-compatible framework.

Usage with FastAPI:
    from fastapi import FastAPI
    from tryaii_dre_sdk.middleware import DREMiddleware

    app = FastAPI()
    app.add_middleware(DREMiddleware, api_key="sk-or-...")

Usage with Starlette:
    from starlette.applications import Starlette
    from tryaii_dre_sdk.middleware import DREMiddleware

    app = Starlette()
    app.add_middleware(DREMiddleware)
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

from tryaii_dre import Router
from tryaii_dre.scoring.priorities import Priorities

logger = logging.getLogger("tryaii_dre_sdk.middleware")


class DREMiddleware:
    """
    ASGI middleware that adds TryAii-DRE routing headers to responses.

    For any request containing a JSON body with a "prompt" or "messages" field,
    the middleware runs the Router to determine the best model and attaches:
      - X-DRE-Model: the recommended model ID
      - X-DRE-Score: the confidence score (0.0 to 1.0)

    This is informational only -- it does not modify the request or redirect
    it to a different backend. Use it for observability and debugging.

    Args:
        app: The ASGI application to wrap.
        priorities: Default routing priorities.
        embedding_model: Sentence-transformers model for embeddings.
        header_prefix: Prefix for response headers (default "X-DRE").
        prompt_field: JSON body field to extract the prompt from.
    """

    def __init__(
        self,
        app,
        priorities: Optional[Priorities] = None,
        embedding_model: Optional[str] = None,
        header_prefix: str = "X-DRE",
        prompt_field: str = "prompt",
    ):
        self.app = app
        self._priorities = priorities
        self._header_prefix = header_prefix
        self._prompt_field = prompt_field

        # Build router
        from tryaii_dre.config import TryaiiDreConfig

        config = TryaiiDreConfig()
        if embedding_model:
            config = TryaiiDreConfig(embedding_model=embedding_model)

        self._router = Router(config=config)

    async def __call__(self, scope, receive, send):
        """ASGI interface."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # Collect the request body to extract the prompt
        body_parts = []
        prompt = None

        async def receive_wrapper():
            message = await receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if body:
                    body_parts.append(body)
            return message

        # Try to extract prompt from request body
        async def send_wrapper(message):
            if message["type"] == "http.response.start" and prompt:
                try:
                    route_result = self._router.route(
                        prompt, priorities=self._priorities, top_k=1
                    )
                    # Inject headers into the response
                    headers = list(message.get("headers", []))
                    headers.append(
                        (
                            f"{self._header_prefix}-Model".lower().encode(),
                            route_result.best_model.encode(),
                        )
                    )
                    headers.append(
                        (
                            f"{self._header_prefix}-Score".lower().encode(),
                            f"{route_result.best_score:.4f}".encode(),
                        )
                    )
                    message = {**message, "headers": headers}
                except Exception:
                    logger.debug("DRE middleware: failed to route prompt", exc_info=True)

            await send(message)

        # We need to intercept the body first
        original_receive = receive

        async def body_interceptor():
            message = await original_receive()
            if message.get("type") == "http.request":
                body = message.get("body", b"")
                if body:
                    body_parts.append(body)
            return message

        # Run the app with body interception, then extract prompt
        # For simplicity, we intercept send to add headers
        received_body = False

        async def smart_receive():
            nonlocal received_body, prompt
            message = await original_receive()
            if message.get("type") == "http.request" and not received_body:
                received_body = True
                body = message.get("body", b"")
                if body:
                    try:
                        data = json.loads(body)
                        # Try "prompt" field first, then extract from "messages"
                        if self._prompt_field in data:
                            prompt = data[self._prompt_field]
                        elif "messages" in data and isinstance(data["messages"], list):
                            # Use the last user message
                            for msg in reversed(data["messages"]):
                                if msg.get("role") == "user":
                                    prompt = msg.get("content", "")
                                    break
                    except (json.JSONDecodeError, KeyError, TypeError):
                        pass
            return message

        await self.app(scope, smart_receive, send_wrapper)
