"""Public Pre-Reasoning V4 API."""
from __future__ import annotations

from .engine import (
    FOCUS_INTERVAL_MINUTES,
    FOCUS_REMINDER,
    FOCUS_SCHEDULER_PROMPT,
    FormError,
    FocusMode,
    MIN_FORM_BLOCKS,
    SHORT_FORM_ALARM,
    V4ReasoningEngine,
    form_spec,
)

__version__ = "4.0.1"

ReasoningEngine = V4ReasoningEngine
ReasoningEngineV4 = V4ReasoningEngine
_ENGINE_CACHE = {}


def get_engine(*, checkpoint_path: str | None = None, device: str = "auto"):
    key = (checkpoint_path, device)
    if key not in _ENGINE_CACHE:
        _ENGINE_CACHE[key] = V4ReasoningEngine(checkpoint_path, device)
    return _ENGINE_CACHE[key]


def get_form() -> dict:
    """Return the structured-form contract without loading the checkpoint."""
    return form_spec()


def analyze_form(
    form_text: str,
    *,
    checkpoint_path: str | None = None,
    device: str = "auto",
) -> dict:
    return get_engine(
        checkpoint_path=checkpoint_path, device=device
    ).analyze_form(form_text)


def pulse(
    form_text: str,
    response: str | None = None,
    *,
    checkpoint_path: str | None = None,
    device: str = "auto",
) -> dict:
    return get_engine(
        checkpoint_path=checkpoint_path, device=device
    ).pulse(form_text, response)


def start_focus_mode(
    *,
    interval_minutes: float = FOCUS_INTERVAL_MINUTES,
    checkpoint_path: str | None = None,
    device: str = "auto",
) -> FocusMode:
    """Start Focus Mode and expose its recurring in-chat scheduler request."""
    return FocusMode(
        interval_minutes=interval_minutes,
        checkpoint_path=checkpoint_path,
        device=device,
    )


def coverage_check(
    form_text: str,
    response: str,
    *,
    checkpoint_path: str | None = None,
    device: str = "auto",
) -> dict:
    """Run the legacy lexical response-coverage check explicitly."""
    return get_engine(
        checkpoint_path=checkpoint_path, device=device
    ).coverage_check(form_text, response)


def coverage_check_result(
    analysis: dict,
    response: str,
    *,
    checkpoint_path: str | None = None,
    device: str = "auto",
) -> dict:
    """Check lexical coverage against an existing analysis result."""
    return get_engine(
        checkpoint_path=checkpoint_path, device=device
    ).coverage_check_result(analysis, response)


def pulse_result(
    analysis: dict,
    response: str,
    *,
    checkpoint_path: str | None = None,
    device: str = "auto",
) -> dict:
    """Compatibility alias for :func:`coverage_check_result`."""
    return coverage_check_result(
        analysis,
        response,
        checkpoint_path=checkpoint_path,
        device=device,
    )


__all__ = [
    "FOCUS_INTERVAL_MINUTES",
    "FOCUS_REMINDER",
    "FOCUS_SCHEDULER_PROMPT",
    "FormError",
    "FocusMode",
    "MIN_FORM_BLOCKS",
    "ReasoningEngine",
    "ReasoningEngineV4",
    "SHORT_FORM_ALARM",
    "__version__",
    "analyze_form",
    "coverage_check",
    "coverage_check_result",
    "get_engine",
    "get_form",
    "pulse",
    "pulse_result",
    "start_focus_mode",
]
