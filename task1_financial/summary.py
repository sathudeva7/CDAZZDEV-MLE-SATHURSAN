"""The Task 1A summary dictionary: the latest price picture in one validated record.

`build_summary` is pure. It reads the prices (with indicator columns), the
momentum signal and Yahoo's fundamentals, and makes no network call, so every
field can be tested offline.

- current_price: the last Close. Prices are dividend- and split-adjusted (see
  data.py), but adjustment only rescales past bars, so the latest Close is the
  quoted price.
- high_52w / low_52w: the highest High and lowest Low of the bars dated on or
  after today - 52 weeks. Being adjusted, a high set before a dividend reads
  slightly below Yahoo's quoted figure, which uses raw prices (for a typical
  dividend payer, well under 1%).
- ytd_return_pct: current price / last Close of the previous calendar year - 1,
  in percent. That close is the standard base: the first close of this year
  would miss the first trading day's move. On adjusted prices this is a total
  return, dividends included.
- pe_ratio: Yahoo's trailingPE; else current price / trailing EPS when EPS is
  positive; else None. Yahoo leaves trailingPE out on negative earnings, where
  a P/E has no meaning.
- indicators: the latest value of each indicator, keyed by the COL_* names.

Every number is rounded to SUMMARY_DECIMALS here and nowhere else; the prices
DataFrame keeps full precision. A field that cannot be computed is None, and
the reason is logged and added to `warnings`, after the warnings from fetching.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1A yfinance data pipeline and summary dictionary', Date: 2026-10-06

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import date

import pandas as pd
from pydantic import BaseModel, Field

from task1_financial.indicators import (
    COL_BB_LOWER,
    COL_BB_PCT_B,
    COL_BB_UPPER,
    COL_MACD,
    COL_MACD_HIST,
    COL_MACD_SIGNAL,
    COL_RSI,
    COL_SMA_LONG,
    COL_SMA_SHORT,
    DEFAULT_PRICE_COLUMN,
)
from task1_financial.signals import MomentumSignal

logger = logging.getLogger(__name__)

FIFTY_TWO_WEEKS = pd.DateOffset(weeks=52)
SUMMARY_DECIMALS = 2
PERCENT = 100.0
HIGH_COLUMN = "High"
LOW_COLUMN = "Low"

# The indicator values reported in the summary, and read by the Task 1B prompt.
SUMMARY_INDICATORS = [
    COL_SMA_SHORT,
    COL_SMA_LONG,
    COL_RSI,
    COL_MACD,
    COL_MACD_SIGNAL,
    COL_MACD_HIST,
    COL_BB_UPPER,
    COL_BB_LOWER,
    COL_BB_PCT_B,
]


class Fundamentals(BaseModel):
    """The fields the summary reads from Yahoo's quote summary (Ticker.info)."""

    company_name: str | None = None
    currency: str | None = None
    trailing_pe: float | None = None
    trailing_eps: float | None = None
    warnings: list[str] = Field(default_factory=list, description="Problems met while fetching")


class MarketSummary(BaseModel):
    """The Task 1A summary for one ticker. Missing values are None, with the reason in `warnings`."""

    ticker: str
    company_name: str | None
    currency: str | None
    as_of: str | None = Field(description="Date of the latest daily bar")
    current_price: float | None
    high_52w: float | None
    low_52w: float | None
    pe_ratio: float | None
    ytd_return_pct: float | None = Field(description="Total return since the last close of the previous year, in percent")
    momentum: MomentumSignal
    indicators: dict[str, float | None]
    warnings: list[str]


def build_summary(
    ticker: str,
    prices: pd.DataFrame,
    signal: MomentumSignal,
    fundamentals: Fundamentals,
    today: date,
    fetch_warnings: Sequence[str] = (),
) -> dict:
    """The summary dictionary for `ticker`, from add_indicators() output and fetched fundamentals."""
    warnings = list(fetch_warnings)

    close = prices[DEFAULT_PRICE_COLUMN].dropna() if DEFAULT_PRICE_COLUMN in prices.columns else pd.Series(dtype=float)
    current = float(close.iloc[-1]) if len(close) else None
    if current is None:
        _warn(warnings, f"{ticker}: no prices, so current price, 52-week range and YTD return are empty")
    high_52w, low_52w = _fifty_two_week_range(prices, current, today, warnings)

    summary = MarketSummary(
        ticker=ticker,
        company_name=fundamentals.company_name,
        currency=fundamentals.currency,
        as_of=close.index[-1].date().isoformat() if len(close) else None,
        current_price=_round(current),
        high_52w=_round(high_52w),
        low_52w=_round(low_52w),
        pe_ratio=_round(_pe_ratio(current, fundamentals, warnings)),
        ytd_return_pct=_round(_ytd_return_pct(close, current, today, warnings)),
        momentum=signal,
        indicators=_latest_indicators(prices),
        warnings=warnings,
    )
    return summary.model_dump()


def _fifty_two_week_range(
    prices: pd.DataFrame, current: float | None, today: date, warnings: list[str]
) -> tuple[float | None, float | None]:
    if current is None:  # already warned: no prices at all
        return None, None
    window = prices.loc[prices.index >= pd.Timestamp(today) - FIFTY_TWO_WEEKS]
    if window.empty:
        _warn(warnings, f"no bars in the 52 weeks to {today}, so the 52-week high and low are empty")
        return None, None
    return float(window[HIGH_COLUMN].max()), float(window[LOW_COLUMN].min())


def _ytd_return_pct(close: pd.Series, current: float | None, today: date, warnings: list[str]) -> float | None:
    if current is None:  # already warned: no prices at all
        return None
    year_start = pd.Timestamp(today.year, 1, 1)
    before = close.loc[close.index < year_start]
    if before.empty or before.iloc[-1] <= 0:
        _warn(warnings, f"no close before {year_start.date()}, so the YTD return is empty")
        return None
    return (current / float(before.iloc[-1]) - 1) * PERCENT


def _pe_ratio(current: float | None, fundamentals: Fundamentals, warnings: list[str]) -> float | None:
    pe, eps = fundamentals.trailing_pe, fundamentals.trailing_eps
    if pe is not None and pe > 0:
        return pe
    if eps is None:
        _warn(warnings, "P/E unavailable: Yahoo returned neither trailingPE nor trailing EPS")
        return None
    if eps <= 0:
        _warn(warnings, f"P/E unavailable: trailing EPS is {eps}, and a P/E on negative earnings is meaningless")
        return None
    if current is None:
        _warn(warnings, "P/E unavailable: Yahoo returned no trailingPE and there is no current price")
        return None
    _warn(warnings, "Yahoo returned no trailingPE, so P/E is computed as current price / trailing EPS")
    return current / eps


def _latest_indicators(prices: pd.DataFrame) -> dict[str, float | None]:
    last = prices.iloc[-1] if len(prices) else pd.Series(dtype=float)
    return {col: _round(last.get(col)) for col in SUMMARY_INDICATORS}


def _round(value) -> float | None:
    """A plain float rounded to SUMMARY_DECIMALS, or None for a missing, NaN or infinite value."""
    if value is None:
        return None
    number = float(value)
    return round(number, SUMMARY_DECIMALS) if math.isfinite(number) else None


def _warn(warnings: list[str], message: str) -> None:
    """Log a fallback and record it in the summary, so the two always agree."""
    logger.warning(message)
    warnings.append(message)
