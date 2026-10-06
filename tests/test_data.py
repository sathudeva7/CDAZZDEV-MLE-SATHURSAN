"""Offline tests for task1_financial/data.py, with yfinance replaced by fakes.

Covers the fetch window, cleaning, the retry loop and the fallbacks. The real
Yahoo call is checked once by tests/test_live_yfinance.py.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1A yfinance data pipeline and summary dictionary', Date: 2026-10-06

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from task1_financial import data
from task1_financial.data import (
    FETCH_RETRIES,
    OHLCV_COLUMNS,
    clean_ohlcv,
    fetch_fundamentals,
    fetch_ohlcv,
)

TODAY = date(2026, 10, 6)


def bars(days: list[str], close: list[float] | None = None) -> pd.DataFrame:
    """OHLCV bars in Yahoo's flat column order (Close first), one per date given."""
    close = close or [100.0 + i for i in range(len(days))]
    return pd.DataFrame(
        {"Close": close, "High": close, "Low": close, "Open": close, "Volume": [1_000] * len(days)},
        index=pd.DatetimeIndex(days, name="Date"),
    )


def daily_bars(start: str, end: str) -> pd.DataFrame:
    days = pd.bdate_range(start, end)
    return bars([d.strftime("%Y-%m-%d") for d in days])


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    """Record the backoff waits instead of sleeping."""
    waited: list[float] = []
    monkeypatch.setattr(data.time, "sleep", waited.append)
    return waited


@pytest.fixture
def downloads(monkeypatch):
    """Replace yf.download with a queue of results; an Exception in the queue is raised."""
    calls: list[dict] = []
    queue: list = []

    def fake_download(ticker, **kwargs):
        calls.append({"ticker": ticker, **kwargs})
        result = queue.pop(0) if queue else pd.DataFrame()
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(data.yf, "download", fake_download)
    return queue, calls


def test_window_is_computed_from_today(downloads, sleeps):
    queue, calls = downloads
    queue.append(daily_bars("2024-10-07", "2026-10-06"))

    history = fetch_ohlcv("AAPL", today=TODAY)

    call = calls[0]
    assert call["end"] == date(2026, 10, 7)  # yfinance's end is exclusive, so today + 1 day
    assert call["start"] == date(2024, 10, 7)  # end - 2 years
    assert call["auto_adjust"] is True
    assert call["multi_level_index"] is False
    assert list(history.prices.columns) == OHLCV_COLUMNS
    assert history.warnings == []


def test_clean_sorts_dedupes_and_drops_incomplete_rows():
    raw = bars(
        ["2026-10-05", "2026-10-01", "2026-10-02", "2026-10-02", "2026-10-03"],
        close=[105.0, 101.0, 102.0, 102.5, 103.0],
    )
    raw.loc[pd.Timestamp("2026-10-03"), "High"] = np.nan
    raw.loc[pd.Timestamp("2026-10-05"), "Volume"] = 0  # a halt day: kept

    prices = clean_ohlcv(raw)

    assert list(prices.index.strftime("%Y-%m-%d")) == ["2026-10-01", "2026-10-02", "2026-10-05"]
    assert prices.loc["2026-10-02", "Close"] == 102.5  # the last of the duplicates
    assert prices.loc["2026-10-05", "Volume"] == 0
    assert list(prices.columns) == OHLCV_COLUMNS


def test_clean_flattens_multiindex_columns_and_drops_timezone():
    raw = bars(["2026-10-01", "2026-10-02"])
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]], names=["Price", "Ticker"])
    raw.index = raw.index.tz_localize("America/New_York")

    prices = clean_ohlcv(raw)

    assert list(prices.columns) == OHLCV_COLUMNS
    assert prices.index.tz is None
    assert prices.index[0] == pd.Timestamp("2026-10-01")


def test_clean_treats_missing_columns_as_empty_with_a_warning():
    warnings: list[str] = []

    prices = clean_ohlcv(bars(["2026-10-01"]).drop(columns=["Volume"]), warnings)

    assert prices.empty and list(prices.columns) == OHLCV_COLUMNS
    assert "missing columns ['Volume']" in warnings[0]


def test_retries_with_backoff_then_succeeds(downloads, sleeps):
    queue, calls = downloads
    queue.extend([ConnectionError("reset by peer"), pd.DataFrame(), daily_bars("2024-10-07", "2026-10-06")])

    history = fetch_ohlcv("AAPL", today=TODAY)

    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]
    assert not history.prices.empty
    assert history.warnings == []


def test_every_attempt_failing_gives_an_empty_frame_and_a_warning(downloads, sleeps):
    _, calls = downloads  # an empty queue: every call returns an empty frame, like a bad ticker

    history = fetch_ohlcv("NOTATICKER", today=TODAY)

    assert len(calls) == FETCH_RETRIES
    assert history.prices.empty
    assert list(history.prices.columns) == OHLCV_COLUMNS
    assert isinstance(history.prices.index, pd.DatetimeIndex)
    assert len(history.warnings) == 1
    assert "no NOTATICKER price history after 3 attempts" in history.warnings[0]


def test_short_history_warns(downloads, sleeps):
    queue, _ = downloads
    queue.append(daily_bars("2026-03-02", "2026-10-06"))  # a recent listing

    history = fetch_ohlcv("NEWCO", today=TODAY)

    assert len(history.prices) > 0
    assert "starts 2026-03-02" in history.warnings[0]


class FakeTicker:
    def __init__(self, info):
        self._info = info

    @property
    def info(self):
        if isinstance(self._info, Exception):
            raise self._info
        return self._info


def test_fundamentals_read_the_info_fields(monkeypatch, sleeps):
    info = {"longName": "Apple Inc.", "currency": "USD", "trailingPE": 38.17546, "trailingEps": 8.72}
    monkeypatch.setattr(data.yf, "Ticker", lambda ticker: FakeTicker(info))

    fundamentals = fetch_fundamentals("AAPL")

    assert fundamentals.company_name == "Apple Inc."
    assert fundamentals.currency == "USD"
    assert fundamentals.trailing_pe == 38.17546
    assert fundamentals.trailing_eps == 8.72
    assert fundamentals.warnings == []


def test_fundamentals_ignore_values_that_are_not_finite_numbers(monkeypatch, sleeps):
    info = {"shortName": "Loss Co", "trailingPE": "Infinity", "trailingEps": float("nan")}
    monkeypatch.setattr(data.yf, "Ticker", lambda ticker: FakeTicker(info))

    fundamentals = fetch_fundamentals("LOSS")

    assert fundamentals.company_name == "Loss Co"  # shortName when longName is missing
    assert fundamentals.trailing_pe is None
    assert fundamentals.trailing_eps is None


def test_fundamentals_failure_gives_empty_fields_and_a_warning(monkeypatch, sleeps):
    monkeypatch.setattr(data.yf, "Ticker", lambda ticker: FakeTicker(RuntimeError("429 Too Many Requests")))

    fundamentals = fetch_fundamentals("AAPL")

    assert fundamentals.company_name is None and fundamentals.trailing_pe is None
    assert sleeps == [1.0, 2.0]
    assert "no AAPL fundamentals after 3 attempts" in fundamentals.warnings[0]
    assert "429 Too Many Requests" in fundamentals.warnings[0]
