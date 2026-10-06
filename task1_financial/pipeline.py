"""Task 1 in two calls: run_market_data() for Task 1A, run_analysis() for Task 1B.

run_market_data() fetches prices, indicators, the momentum signal,
fundamentals, the summary and headlines, and needs no API keys.
run_analysis() adds headline sentiment, the Sentiment score and the
Recommendation, and needs the LLM key (Jev's key is optional: without it the
LLM labels every headline). They are separate so Task 1A runs without keys.
This module only composes: the fetching lives in data.py and news.py, the
indicators in indicators.py and signals.py, the summary fields in summary.py,
and the model calls in sentiment.py and recommendation.py.

Headlines sit beside the summary, not inside it, because the brief fixes the
summary dictionary's fields. Their warnings still go into summary['warnings'],
so every fallback is listed in one place.

`today` defaults to the current date in New York, where the default ticker
trades, rather than the machine's date: Colab runs on UTC, which is already
tomorrow during the New York evening.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1A yfinance data pipeline and summary dictionary', Date: 2026-10-06
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1A step 2, news headlines, as designed in the grilling rounds', Date: 2026-10-06
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from common.jev import JevClient
from common.llm import StructuredLLM
from task1_financial.data import fetch_fundamentals, fetch_ohlcv
from task1_financial.indicators import add_indicators
from task1_financial.news import Headline, fetch_headlines
from task1_financial.recommendation import RecommendationResult, recommend
from task1_financial.schemas import HeadlineSentiment, SentimentSummary
from task1_financial.sentiment import score_headlines, summarise_sentiment
from task1_financial.signals import momentum_signal
from task1_financial.summary import build_summary

logger = logging.getLogger(__name__)

DEFAULT_TICKER = "AAPL"
MARKET_TIMEZONE = ZoneInfo("America/New_York")
# Every LLM and Jev request of a run is logged here, one JSON line each.
LOG_DIR = Path(__file__).parent / "logs"


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


@dataclass
class Analysis:
    headline_sentiments: list[HeadlineSentiment]  # same order as MarketData.headlines
    sentiment: SentimentSummary  # the Sentiment score
    recommendation: RecommendationResult
    warnings: list[str] = field(default_factory=list)


def run_analysis(market: MarketData, llm: StructuredLLM | None = None, jev: JevClient | None = None) -> Analysis:
    """Task 1B on top of run_market_data()'s output. Never raises for a model failure; see `warnings`.

    The clients are created here, not at import, because StructuredLLM needs
    its key the moment it is built. Tests pass fakes instead.
    """
    llm = llm or StructuredLLM(log_dir=LOG_DIR)
    jev = jev or JevClient(log_dir=LOG_DIR)
    summary = market.summary
    warnings: list[str] = []

    items = score_headlines(market.headlines, summary["ticker"], summary.get("company_name"), jev, llm)
    sentiment = summarise_sentiment(items, warnings)
    if sentiment.by_llm:
        _warn(warnings, f"Jev was unavailable for {sentiment.by_llm} headlines, so the LLM labelled them")
    if sentiment.failed:
        _warn(warnings, f"{sentiment.failed} headlines could not be scored and are left out of the Sentiment score")

    result = recommend(market.prices, summary, sentiment, llm)
    if not result.ok:
        _warn(warnings, "the Recommendation is a placeholder Hold: no LLM provider answered")
    return Analysis(headline_sentiments=items, sentiment=sentiment, recommendation=result, warnings=warnings)


def _warn(warnings: list[str], message: str) -> None:
    """Log a fallback and record it for the analysis, so the two always agree."""
    logger.warning(message)
    warnings.append(message)
