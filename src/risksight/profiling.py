"""Opt-in, low-overhead timing hooks for the unchanged RiskSight pipeline."""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter
from typing import Iterator


_ACTIVE_TIMINGS: ContextVar[dict[str, list[float]] | None] = ContextVar(
    "risksight_active_timings", default=None
)


@contextmanager
def collect_stage_timings() -> Iterator[dict[str, list[float]]]:
    """Collect stage durations in seconds for calls made in this context."""
    timings: dict[str, list[float]] = defaultdict(list)
    token = _ACTIVE_TIMINGS.set(timings)
    try:
        yield timings
    finally:
        _ACTIVE_TIMINGS.reset(token)


@contextmanager
def timed_stage(name: str) -> Iterator[None]:
    """Record a stage only when a benchmark has enabled collection."""
    timings = _ACTIVE_TIMINGS.get()
    if timings is None:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        timings[name].append(perf_counter() - start)
