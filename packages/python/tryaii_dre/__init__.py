"""
TryAii-DRE -- Embedding-based AI Model Router

Understands your prompt semantically and routes to the best model
based on benchmarks, cost, speed, and quality priorities.

Usage:
    from tryaii_dre import Router

    router = Router()
    result = router.route("Write a Python function to merge sorted arrays")
    print(result.best_model)
    print(result.scores)
"""

import logging

logging.getLogger("tryaii_dre").addHandler(logging.NullHandler())

from tryaii_dre.router import Router, RouteResult
from tryaii_dre.registry.models import ModelRegistry, ModelInfo
from tryaii_dre.scoring.priorities import Priorities, DEFAULT_PRIORITIES
from tryaii_dre.benchmarks.registry import BenchmarkRegistry
from tryaii_dre.config import TryaiiDreConfig

__version__ = "0.1.0"

__all__ = [
    "Router",
    "RouteResult",
    "ModelRegistry",
    "ModelInfo",
    "Priorities",
    "DEFAULT_PRIORITIES",
    "BenchmarkRegistry",
    "TryaiiDreConfig",
    "__version__",
]
