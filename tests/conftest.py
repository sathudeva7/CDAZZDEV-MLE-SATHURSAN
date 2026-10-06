"""Fixtures shared by the Task 3 tests: fake prices, no backoff sleeps, and a fake web search.

Named apart from test_data.py's own `downloads` and `sleeps` fixtures, which
behave differently, so neither shadows the other.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import pytest

from task1_financial import data
from tests.fakes import FakeSearch, two_years_of_bars


@pytest.fixture
def no_sleep(monkeypatch):
    """Record the retry backoff waits instead of sleeping."""
    waited: list[float] = []
    monkeypatch.setattr(data.time, "sleep", waited.append)
    return waited


@pytest.fixture
def fake_prices(monkeypatch, no_sleep):
    """Replace yf.download; `frames` holds what each download returns (the last one repeats)."""
    calls: list[str] = []
    frames = [two_years_of_bars()]

    def fake_download(ticker, **_):
        calls.append(ticker)
        return frames[min(len(calls), len(frames)) - 1]

    monkeypatch.setattr(data.yf, "download", fake_download)
    return type("Downloads", (), {"calls": calls, "frames": frames})


@pytest.fixture
def fake_search():
    """The FakeSearch class with no replies set; tests fill fake_search.replies per backend."""
    FakeSearch.replies, FakeSearch.queries = {}, []
    return FakeSearch
