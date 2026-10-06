"""Persistent memory: the final research brief saved as JSON, keyed by ticker and date.

    task3_agentic/cache/{TICKER}_{YYYY-MM-DD}.json

The date is the New York market date of the run (ToolSession.today), so a
run after the close and one the next morning use different files. A file
counts as a hit only when it parses and validates against the current
schemas and SCHEMA_VERSION; a missing, corrupt or outdated file is a miss,
logged with its reason, and the pipeline runs in full. Every lookup and
save writes a `cache` event to agent_trace.jsonl. The pipeline never saves a
report written from the template (no model answered), so the next run retries.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 3: the 3B two-agent pipeline with the critique loop and the persistent cache, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, ValidationError

from task3_agentic.schemas import (
    ClarificationRequest,
    ClarificationResponse,
    DataBrief,
    ResearchReport,
)
from task3_agentic.trace import TraceLog

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "cache"
# Bump when a cached schema changes shape, so old files are treated as misses rather than misread.
SCHEMA_VERSION = 1


class CachedRun(BaseModel):
    """Everything a later run needs to show the result without calling a tool."""

    schema_version: int
    ticker: str
    as_of: date
    run_id: str
    profile: str
    created_at: datetime
    report: ResearchReport
    brief: DataBrief
    request: ClarificationRequest
    response: ClarificationResponse


def cache_path(cache_dir: Path | str, ticker: str, day: date) -> Path:
    return Path(cache_dir) / f"{ticker.upper()}_{day.isoformat()}.json"


def load_cached(cache_dir: Path | str, ticker: str, day: date, trace: TraceLog) -> CachedRun | None:
    """The cached run for (ticker, day), or None for a miss. Never raises for a bad file."""
    path = cache_path(cache_dir, ticker, day)
    if not path.exists():
        return _miss(trace, path, "no file")
    try:
        cached = CachedRun.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, ValueError, OSError) as exc:
        return _miss(trace, path, f"unreadable: {type(exc).__name__}", warn=True)
    if cached.schema_version != SCHEMA_VERSION:
        return _miss(trace, path, f"schema version {cached.schema_version}, expected {SCHEMA_VERSION}", warn=True)
    logger.info("cache hit: %s (saved %s)", path, cached.created_at)
    trace.write("cache", action="hit", path=str(path), saved_at=cached.created_at.isoformat(), saved_run_id=cached.run_id)
    return cached


def save_cached(cache_dir: Path | str, run: CachedRun, trace: TraceLog) -> Path:
    path = cache_path(cache_dir, run.ticker, run.as_of)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    logger.info("saved the research brief to %s", path)
    trace.write("cache", action="save", path=str(path))
    return path


def _miss(trace: TraceLog, path: Path, reason: str, *, warn: bool = False) -> None:
    (logger.warning if warn else logger.info)("cache miss for %s: %s", path, reason)
    trace.write("cache", action="miss", path=str(path), reason=reason)
