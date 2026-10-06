"""Daily prices and fundamentals from Yahoo Finance, through yfinance.

This is the only module in the repo that talks to Yahoo. It never raises for
a data problem: a failed or empty fetch is retried, then becomes an empty
result plus a warning, so the pipeline still runs to completion.

- Window: computed from `today`, never written as a date string. yfinance's
  `end` is exclusive, so end = today + 1 day and start = end - LOOKBACK_YEARS.
- Prices: `auto_adjust=True` (yfinance's default, passed explicitly anyway).
  Every past bar is rescaled for splits and dividends, so a dividend does not
  show up as a price drop in the indicators, and there is no Adj Close column.
- Cleaning, for task1_financial/indicators.py: flat columns, a timezone-naive
  date index in ascending order, duplicate dates resolved by keeping the last
  row, and rows with a missing Open/High/Low/Close dropped. Zero-volume rows
  (trading halts) and today's unfinished bar are kept: no indicator reads
  volume, and the current price should be current. Each step logs what it
  removed.
- Retries: FETCH_RETRIES attempts on an exception or an empty result, waiting
  FETCH_BACKOFF_SECONDS and doubling each time. Errors are not classified,
  because yfinance's exception types change between versions; a mistyped
  ticker costs a few seconds of retries.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1A yfinance data pipeline and summary dictionary', Date: 2026-10-06

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import TypeVar

import pandas as pd
import yfinance as yf

from task1_financial.summary import Fundamentals

logger = logging.getLogger(__name__)

LOOKBACK_YEARS = 2
# Weekends and holidays can push the first bar a few days past the requested
# start; a bigger gap means the ticker has less history than asked for.
HISTORY_START_TOLERANCE_DAYS = 7
FETCH_RETRIES = 3
FETCH_BACKOFF_SECONDS = 1.0

PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
OHLCV_COLUMNS = [*PRICE_COLUMNS, "Volume"]
INDEX_NAME = "Date"

T = TypeVar("T")


@dataclass
class PriceHistory:
    """Cleaned daily bars, the window that was requested, and any problems met."""

    prices: pd.DataFrame
    start: date
    end: date  # exclusive, as yfinance treats it
    warnings: list[str] = field(default_factory=list)


def fetch_ohlcv(ticker: str, today: date) -> PriceHistory:
    """Cleaned daily OHLCV bars for the LOOKBACK_YEARS up to and including `today`."""
    end = today + timedelta(days=1)
    start = (pd.Timestamp(end) - pd.DateOffset(years=LOOKBACK_YEARS)).date()
    warnings: list[str] = []

    raw = _with_retries(
        lambda: yf.download(
            ticker, start=start, end=end, auto_adjust=True, multi_level_index=False, progress=False
        ),
        what=f"{ticker} price history",
        is_empty=lambda frame: frame is None or frame.empty,
        warnings=warnings,
    )
    prices = clean_ohlcv(raw if raw is not None else _empty_prices(), warnings)
    if prices.empty:
        return PriceHistory(prices, start, end, warnings)

    first, last = prices.index[0].date(), prices.index[-1].date()
    logger.info("%s: %d daily bars from %s to %s", ticker, len(prices), first, last)
    if (first - start).days > HISTORY_START_TOLERANCE_DAYS:
        _warn(warnings, f"{ticker} history starts {first}, later than the requested {start}: less than {LOOKBACK_YEARS} years of data")
    return PriceHistory(prices, start, end, warnings)


def clean_ohlcv(raw: pd.DataFrame, warnings: list[str] | None = None) -> pd.DataFrame:
    """Raw yfinance bars in the shape add_indicators() expects (see the module docstring)."""
    warnings = [] if warnings is None else warnings
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        # yf.download keeps a (Price, Ticker) column level unless multi_level_index=False.
        frame.columns = frame.columns.get_level_values(0)
    missing = [col for col in OHLCV_COLUMNS if col not in frame.columns]
    if missing:
        if not frame.empty:
            _warn(warnings, f"price data is missing columns {missing}, so it is treated as empty")
        return _empty_prices()

    frame = frame[OHLCV_COLUMNS]
    frame.index = pd.DatetimeIndex(frame.index, name=INDEX_NAME)
    if frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)  # keeps the exchange's calendar date
    frame = frame.sort_index()

    duplicates = frame.index.duplicated(keep="last")
    if duplicates.any():
        logger.info("dropped %d duplicate date rows, keeping the last of each", duplicates.sum())
        frame = frame[~duplicates]
    incomplete = frame[PRICE_COLUMNS].isna().any(axis=1)
    if incomplete.any():
        logger.info("dropped %d rows with a missing open, high, low or close", incomplete.sum())
        frame = frame[~incomplete]
    return frame


def fetch_fundamentals(ticker: str) -> Fundamentals:
    """Company name, currency, trailing P/E and EPS from Yahoo; empty fields when unavailable."""
    warnings: list[str] = []
    info = _with_retries(
        lambda: yf.Ticker(ticker).info,
        what=f"{ticker} fundamentals",
        is_empty=lambda result: not result,
        warnings=warnings,
    ) or {}
    return Fundamentals(
        company_name=_as_text(info.get("longName")) or _as_text(info.get("shortName")),
        currency=_as_text(info.get("currency")),
        trailing_pe=_as_number(info.get("trailingPE")),
        trailing_eps=_as_number(info.get("trailingEps")),
        warnings=warnings,
    )


def _with_retries(fetch: Callable[[], T], what: str, is_empty: Callable[[T], bool], warnings: list[str]) -> T | None:
    """`fetch()`'s first non-empty result, trying FETCH_RETRIES times; None and a warning if all fail."""
    problem = "empty result"
    for attempt in range(FETCH_RETRIES):
        if attempt:
            wait = FETCH_BACKOFF_SECONDS * 2 ** (attempt - 1)
            logger.info("retrying %s in %.0fs (attempt %d of %d): %s", what, wait, attempt + 1, FETCH_RETRIES, problem)
            time.sleep(wait)
        try:
            result = fetch()
        # yfinance raises network, HTTP and parsing errors of many unrelated types, and
        # a failed fetch must become a warning, never a crash.
        except Exception as exc:  # noqa: BLE001
            problem = f"{type(exc).__name__}: {exc}"
            continue
        if not is_empty(result):
            return result
        problem = "empty result"
    _warn(warnings, f"no {what} after {FETCH_RETRIES} attempts (last problem: {problem}); check the ticker symbol or try later")
    return None


def _empty_prices() -> pd.DataFrame:
    return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], name=INDEX_NAME), dtype=float)


def _as_number(value) -> float | None:
    """A finite float, or None for anything else (Yahoo sometimes sends 'Infinity' or NaN)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _as_text(value) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _warn(warnings: list[str], message: str) -> None:
    """Log a fallback and record it for the summary, so the two always agree."""
    logger.warning(message)
    warnings.append(message)
