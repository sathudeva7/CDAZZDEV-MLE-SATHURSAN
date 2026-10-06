"""Task 1A in one call: prices, indicators, momentum signal, fundamentals, the summary and headlines.

The notebook and Task 3's tools call run_market_data(). Headline sentiment
(Task 1B) will join here. This module only composes: the fetching lives in
data.py and news.py, the indicators in indicators.py and signals.py, and the
summary fields in summary.py.

Headlines sit beside the summary, not inside it, because the brief fixes the
summary dictionary's fields. Their warnings still go into summary['warnings'],
so every fallback is listed in one place.

`today` defaults to the current date in New York, where the default ticker
trades, rather than the machine's date: Colab runs on UTC, which is already
tomorrow during the New York evening.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1A yfinance data pipeline and summary dictionary', Date: 2026-10-06
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1A step 2, news headlines, as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from task1_financial.data import fetch_fundamentals, fetch_ohlcv
from task1_financial.indicators import add_indicators
from task1_financial.news import Headline, fetch_headlines
from task1_financial.signals import momentum_signal
from task1_financial.summary import build_summary

DEFAULT_TICKER = "AAPL"
MARKET_TIMEZONE = ZoneInfo("America/New_York")


@dataclass
class MarketData:
    prices: pd.DataFrame  # daily bars with every indicator column added
    summary: dict  # MarketSummary.model_dump(): the Task 1A summary dictionary
    headlines: list[Headline]  # recent news, newest first; may be empty (see summary['warnings'])


def run_market_data(ticker: str = DEFAULT_TICKER, today: date | None = None) -> MarketData:
    """Fetch, clean and analyse `ticker`. Never raises for a data problem; see summary['warnings']."""
    ticker = ticker.strip().upper()
    today = today or datetime.now(MARKET_TIMEZONE).date()

    history = fetch_ohlcv(ticker, today)
    prices = add_indicators(history.prices)
    signal = momentum_signal(prices)
    fundamentals = fetch_fundamentals(ticker)
    news = fetch_headlines(ticker, today)
    summary = build_summary(
        ticker,
        prices,
        signal,
        fundamentals,
        today=today,
        fetch_warnings=[*history.warnings, *fundamentals.warnings, *news.warnings],
    )
    return MarketData(prices=prices, summary=summary, headlines=news.headlines)
