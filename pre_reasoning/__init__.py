"""Public package API for Pre-Reasoning."""

from __future__ import annotations

from typing import Optional

from .engine_core import ReasoningEngineV25 as ReasoningEngineV25Legacy
from .engine import ReasoningEngineV252
from .inference import ReasoningEngineV3

ReasoningEngine = ReasoningEngineV252
ReasoningEngineV25 = ReasoningEngineV252

__all__ = [
    "ReasoningEngineV25",
    "ReasoningEngineV25Legacy",
    "ReasoningEngineV252",
    "ReasoningEngineV3",
    "ReasoningEngine",
    "analyze",
    "pulse",
    "get_engine",
]


def get_engine(
    *,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
) -> ReasoningEngineV25:
    """Create a reasoning engine (13.7M neural perception + graph analysis)."""
    return ReasoningEngine(
        checkpoint_path=checkpoint_path,
        device=device,
    )


def analyze(
    text: str,
    *,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
) -> dict:
    """Analyze problem text and return a structural trace result."""
    return get_engine(
        checkpoint_path=checkpoint_path,
        device=device,
    ).analyze(text)


def pulse(
    original_problem: str,
    response: str,
    *,
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
) -> dict:
    """Check whether a draft response addresses detected root blockers."""
    return get_engine(
        checkpoint_path=checkpoint_path,
        device=device,
    ).pulse(original_problem, response)
