"""One live call to Yahoo, to confirm the facts the offline tests assume.

Skipped by default (see pytest.ini). Run it with: pytest -m live
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1A yfinance data pipeline and summary dictionary', Date: 2026-10-06

from __future__ import annotations

import pandas as pd
import pytest
import yfinance as yf

from task1_financial.data import LOOKBACK_YEARS, OHLCV_COLUMNS
from task1_financial.pipeline import DEFAULT_TICKER, run_market_data

# A year has about 252 trading days; allow for holidays at either end of the window.
MIN_BARS = 245 * LOOKBACK_YEARS
# Adjusted highs and lows sit slightly below Yahoo's raw-price quotes when a
# dividend falls inside the window (see summary.py).
QUOTE_TOLERANCE = 0.01


@pytest.mark.live
def test_live_run_for_the_default_ticker():
    result = run_market_data(DEFAULT_TICKER)
    prices, summary = result.prices, result.summary

    assert list(prices.columns[: len(OHLCV_COLUMNS)]) == OHLCV_COLUMNS
    assert len(prices) >= MIN_BARS
    assert prices.index.is_monotonic_increasing and prices.index.is_unique
    span = prices.index[-1] - prices.index[0]
    assert span >= pd.Timedelta(days=365 * LOOKBACK_YEARS - 7)

    for field in ["current_price", "high_52w", "low_52w", "pe_ratio", "ytd_return_pct"]:
        assert summary[field] is not None, field
    assert summary["momentum"]["unavailable"] == []

    quoted = yf.Ticker(DEFAULT_TICKER).fast_info
    assert summary["high_52w"] == pytest.approx(quoted.year_high, rel=QUOTE_TOLERANCE)
    assert summary["low_52w"] == pytest.approx(quoted.year_low, rel=QUOTE_TOLERANCE)
