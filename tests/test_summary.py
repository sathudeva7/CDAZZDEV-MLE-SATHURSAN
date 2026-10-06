"""Tests for task1_financial/summary.py, on frames small enough to check by hand.

The main frame has `today` = 2026-01-15, so:
- 52 weeks back is 2025-01-16. The bar on 2025-01-15 (High 500, Low 1) is one
  day outside the window and must be ignored; the bar on 2025-01-16 is inside.
- The last close of 2025 is 100 and the current price is 110, so YTD is +10%.
  Using the first close of 2026 (102) instead would give 7.84%.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1A yfinance data pipeline and summary dictionary', Date: 2026-10-06

from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

from task1_financial.indicators import COL_RSI, COL_SMA_LONG, add_indicators
from task1_financial.signals import momentum_signal
from task1_financial.summary import SUMMARY_INDICATORS, Fundamentals, build_summary

TODAY = date(2026, 1, 15)

# (date, high, low, close)
BARS = [
    ("2025-01-15", 500.0, 1.0, 90.0),  # outside the 52-week window by one day
    ("2025-01-16", 120.0, 80.0, 95.0),  # first day inside the window
    ("2025-06-02", 130.0, 90.0, 120.0),  # the 52-week high
    ("2025-12-31", 105.0, 95.0, 100.0),  # last close of the previous year: the YTD base
    ("2026-01-02", 104.0, 100.0, 102.0),
    ("2026-01-14", 112.0, 108.0, 110.0),  # the current price
]
FUNDAMENTALS = Fundamentals(company_name="Example Corp", currency="USD", trailing_pe=38.17546, trailing_eps=2.88)


def prices_from(bars) -> pd.DataFrame:
    """Cleaned OHLCV bars with every indicator column added (mostly NaN, as the frame is short)."""
    frame = pd.DataFrame(
        {
            "Open": [close for _, _, _, close in bars],
            "High": [high for _, high, _, _ in bars],
            "Low": [low for _, _, low, _ in bars],
            "Close": [close for _, _, _, close in bars],
            "Volume": [1_000] * len(bars),
        },
        index=pd.DatetimeIndex([day for day, *_ in bars], name="Date"),
    )
    return add_indicators(frame)


def summarise(prices: pd.DataFrame, fundamentals: Fundamentals = FUNDAMENTALS, **kwargs) -> dict:
    return build_summary("EXMP", prices, momentum_signal(prices), fundamentals, today=TODAY, **kwargs)


def test_hand_checked_fields():
    summary = summarise(prices_from(BARS))

    assert summary["ticker"] == "EXMP"
    assert summary["company_name"] == "Example Corp"
    assert summary["currency"] == "USD"
    assert summary["as_of"] == "2026-01-14"
    assert summary["current_price"] == 110.0
    assert summary["high_52w"] == 130.0  # not 500: that bar is outside the window
    assert summary["low_52w"] == 80.0  # not 1, for the same reason
    assert summary["ytd_return_pct"] == 10.0  # 110 / 100 - 1, not 110 / 102 - 1
    assert summary["pe_ratio"] == 38.18  # Yahoo's trailingPE, rounded
    assert summary["momentum"]["as_of"] == "2026-01-14"


def test_every_required_field_is_present_and_json_serialisable():
    summary = summarise(prices_from(BARS))

    for field in ["current_price", "high_52w", "low_52w", "pe_ratio", "ytd_return_pct", "momentum"]:
        assert field in summary
    assert set(summary["indicators"]) == set(SUMMARY_INDICATORS)
    json.dumps(summary)  # no numpy floats or Timestamps left


def test_indicators_are_the_latest_values_rounded():
    prices = prices_from(BARS)
    prices.loc[prices.index[-1], COL_RSI] = 61.23456

    summary = summarise(prices)

    assert summary["indicators"][COL_RSI] == 61.23
    assert summary["indicators"][COL_SMA_LONG] is None  # 6 bars is too few for SMA-200


def test_ytd_is_empty_without_a_bar_before_new_year():
    this_year_only = [bar for bar in BARS if bar[0].startswith("2026")]

    summary = summarise(prices_from(this_year_only))

    assert summary["ytd_return_pct"] is None
    assert any("YTD" in w for w in summary["warnings"])


def test_ytd_is_zero_before_the_first_trading_day_of_the_year():
    last_year_only = [bar for bar in BARS if bar[0].startswith("2025")]

    summary = summarise(prices_from(last_year_only))

    assert summary["current_price"] == 100.0
    assert summary["ytd_return_pct"] == 0.0


def test_pe_falls_back_to_price_over_eps():
    no_pe = Fundamentals(trailing_pe=None, trailing_eps=11.0)

    summary = summarise(prices_from(BARS), no_pe)

    assert summary["pe_ratio"] == 10.0  # 110 / 11
    assert any("trailing EPS" in w for w in summary["warnings"])


@pytest.mark.parametrize(
    ("fundamentals", "reason"),
    [
        (Fundamentals(trailing_pe=None, trailing_eps=-2.0), "negative earnings"),
        (Fundamentals(trailing_pe=None, trailing_eps=0.0), "negative earnings"),
        (Fundamentals(), "neither trailingPE nor trailing EPS"),
    ],
)
def test_pe_is_empty_with_a_reason(fundamentals, reason):
    summary = summarise(prices_from(BARS), fundamentals)

    assert summary["pe_ratio"] is None
    assert any(reason in w for w in summary["warnings"])


def test_no_bars_in_the_last_52_weeks():
    stale = [bar for bar in BARS if bar[0] <= "2025-01-15"]

    summary = summarise(prices_from(stale))

    assert summary["high_52w"] is None and summary["low_52w"] is None
    assert any("52-week" in w for w in summary["warnings"])


def test_all_missing_inputs_give_none_without_raising():
    empty = prices_from([])

    summary = summarise(empty, Fundamentals(), fetch_warnings=["no EXMP price history after 3 attempts"])

    for field in ["company_name", "currency", "as_of", "current_price", "high_52w", "low_52w", "pe_ratio", "ytd_return_pct"]:
        assert summary[field] is None, field
    assert all(value is None for value in summary["indicators"].values())
    assert summary["momentum"]["label"] == "Neutral"
    # Fetch warnings come first, then one line per field the summary could not fill.
    assert summary["warnings"][0] == "no EXMP price history after 3 attempts"
    assert any("no prices" in w for w in summary["warnings"])
    json.dumps(summary)


def test_warnings_are_logged_too(caplog):
    prices = prices_from(BARS)  # outside the capture: the short frame logs indicator warnings of its own

    with caplog.at_level("WARNING", logger="task1_financial.summary"):
        summary = summarise(prices, Fundamentals())

    logged = [r.getMessage() for r in caplog.records if r.name == "task1_financial.summary"]
    assert summary["warnings"]
    assert summary["warnings"] == logged
