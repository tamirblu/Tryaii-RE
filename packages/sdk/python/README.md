# TryAii-DRE SDK

High-level Python SDK for TryAii-DRE (Differential Routing Engine). Provides a
unified `DREClient` that wraps the core Router and OpenRouter integration into a
single, easy-to-use interface with async support and framework middleware.

## Installation

```bash
pip install tryaii-dre-sdk
```

For FastAPI/Starlette middleware support:

```bash
pip install tryaii-dre-sdk[fastapi]
```

## Quick Start

```python
from tryaii_dre_sdk import DREClient

client = DREClient(api_key="sk-or-...")

# Route a prompt and get an AI response in one call
response = client.chat("Write a Python quicksort implementation")
print(response.content)
print(response.model_used)

# Just route (no API call) to see which model would be selected
result = client.route("Explain quantum computing")
print(result.best_model)
print(result.scores[:3])

# Route and chat -- returns both the routing decision and the response
route_result, response = client.route_and_chat("Translate this to French: Hello world")
print(route_result.best_model)
print(response.content)
```

## Async Support

```python
from tryaii_dre_sdk.async_client import AsyncDREClient

client = AsyncDREClient(api_key="sk-or-...")

response = await client.chat("Write a REST API in FastAPI")
print(response.content)

# Async streaming
async for chunk in client.stream("Explain machine learning"):
    print(chunk, end="")
```

## Custom Priorities

Control the quality/cost/speed tradeoff:

```python
from tryaii_dre import Priorities

client = DREClient(api_key="sk-or-...", priorities=Priorities.performance())

# Or per-request:
response = client.chat(
    "Optimize this SQL query",
    priorities=Priorities(quality=5, cost=1, speed=2),
)
```

## ASGI Middleware

Add DRE routing headers to your FastAPI or Starlette application:

```python
from fastapi import FastAPI
from tryaii_dre_sdk.middleware import DREMiddleware

app = FastAPI()
app.add_middleware(DREMiddleware, api_key="sk-or-...")

# Every response now includes X-DRE-Model and X-DRE-Score headers
```

## API Reference

### DREClient

| Method | Description |
|---|---|
| `chat(prompt, ...)` | Route to best model and return the response |
| `stream(prompt, ...)` | Route and stream the response as chunks |
| `route(prompt, ...)` | Route only -- returns RouteResult, no API call |
| `route_and_chat(prompt, ...)` | Returns both RouteResult and response |

### AsyncDREClient

Same interface as `DREClient`, but all methods are async/await.

## License

Apache 2.0
