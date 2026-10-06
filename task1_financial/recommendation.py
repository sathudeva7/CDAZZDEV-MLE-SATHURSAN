"""The Recommendation: the LLM's Buy, Hold or Sell call over the next 90 days.

The LLM gets facts, not a verdict. Each fact pairs a raw value with a
relationship computed here (how far price is from each moving average,
whether the trend gap is widening, which way RSI and the MACD histogram are
moving), because a model reasons well over relationships but can get the
arithmetic wrong. The rule-based Momentum signal is deliberately left out, so
the model has to combine the indicators itself rather than copy our label;
the notebook shows the two side by side. P/E is left out as well: this call is
technical analysis plus news.

A value that cannot be computed (a young ticker with no 200-day average, or
no prices at all) becomes "unavailable", and the prompt forbids guessing it.
If every provider fails, the result is a Hold marked ok=False.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from common.llm import StructuredLLM
from task1_financial.indicators import (
    COL_BB_PCT_B,
    COL_MACD,
    COL_MACD_HIST,
    COL_MACD_SIGNAL,
    COL_RSI,
    COL_SMA_LONG,
    COL_SMA_SHORT,
    COL_VOLATILITY,
    DEFAULT_PRICE_COLUMN,
    RSI_PERIOD,
    SMA_LONG_WINDOW,
    SMA_SHORT_WINDOW,
    VOLATILITY_WINDOW,
)
from task1_financial.news import NEWS_MAX_AGE_DAYS
from task1_financial.prompts import RECOMMENDATION
from task1_financial.schemas import Recommendation, SentimentSummary
from task1_financial.signals import RSI_OVERBOUGHT, RSI_OVERSOLD

# Matches Task 3's question ("risks over the next 90 days"), and suits indicators
# that run from a 14-day RSI to a 200-day average.
RECOMMENDATION_HORIZON_DAYS = 90
RSI_LOOKBACK_DAYS = 5
TREND_LOOKBACK_DAYS = 10
PERCENT = 100.0
UNAVAILABLE = "unavailable"
# Weighing several indicators against each other is real reasoning, so medium effort.
RECOMMENDATION_EFFORT = "medium"

FALLBACK_RECOMMENDATION = Recommendation(
    recommendation="Hold",
    justification=(
        "No language model was reachable, so this is a placeholder rather than an analysis. "
        "Hold is shown because it commits to neither direction. "
        "Read the Momentum signal and the indicator values directly instead."
    ),
    key_factors=["Placeholder: the LLM call failed", "Not a judgement on the indicators"],
)


@dataclass(frozen=True)
class RecommendationResult:
    """The Recommendation, whether the LLM produced it, and the facts it was given."""

    value: Recommendation
    ok: bool  # False when `value` is FALLBACK_RECOMMENDATION
    provider: str | None
    facts: list[str]


def recommend(prices: pd.DataFrame, summary: dict, sentiment: SentimentSummary, llm: StructuredLLM) -> RecommendationResult:
    """Ask the LLM for a Buy, Hold or Sell call on the facts. Never raises for a model failure."""
    technical = technical_facts(prices, summary)
    news = sentiment_facts(sentiment)
    result = llm.call(
        RECOMMENDATION,
        {
            "company": summary.get("company_name") or summary.get("ticker"),
            "ticker": summary.get("ticker"),
            "as_of": summary.get("as_of") or UNAVAILABLE,
            "horizon_days": RECOMMENDATION_HORIZON_DAYS,
            "technical_facts": "\n".join(technical),
            "sentiment_facts": "\n".join(news),
        },
        Recommendation,
        fallback=FALLBACK_RECOMMENDATION,
        reasoning_effort=RECOMMENDATION_EFFORT,
    )
    return RecommendationResult(value=result.value, ok=result.ok, provider=result.provider, facts=technical + news)


def technical_facts(prices: pd.DataFrame, summary: dict) -> list[str]:
    """One line per fact, each a raw value with its computed relationship, or 'unavailable'."""
    close = _at(prices, DEFAULT_PRICE_COLUMN)
    sma_short, sma_long = _at(prices, COL_SMA_SHORT), _at(prices, COL_SMA_LONG)
    currency = f" {summary['currency']}" if summary.get("currency") else ""
    return [
        f"Close: {close:.2f}{currency}" if close is not None else f"Close: {UNAVAILABLE}",
        _versus(f"Close vs {SMA_SHORT_WINDOW}-day SMA", close, sma_short),
        _versus(f"Close vs {SMA_LONG_WINDOW}-day SMA", close, sma_long),
        _trend_gap(prices),
        _rsi(prices),
        _macd(prices),
        _bollinger(prices),
        _range_position(close, summary.get("low_52w"), summary.get("high_52w")),
        _labelled(f"{VOLATILITY_WINDOW}-day annualised volatility", _at(prices, COL_VOLATILITY), lambda v: f"{v * PERCENT:.1f}%"),
        _labelled("Year-to-date return", summary.get("ytd_return_pct"), lambda v: f"{v:+.1f}%"),
    ]


def sentiment_facts(sentiment: SentimentSummary) -> list[str]:
    if sentiment.score is None:
        return [f"Sentiment score: {UNAVAILABLE} (no headline could be scored)"]
    counts = sentiment.counts
    return [
        (
            f"Sentiment score: {sentiment.score:+.2f} ({sentiment.label}) on a scale from -1 to +1, "
            f"from {sentiment.scored} headlines of the last {NEWS_MAX_AGE_DAYS} days "
            f"({counts.get('positive', 0)} positive, {counts.get('negative', 0)} negative, {counts.get('neutral', 0)} neutral)"
        )
    ]


# --- one fact each ---------------------------------------------------------


def _versus(label: str, value: float | None, reference: float | None) -> str:
    if value is None or reference is None:
        return f"{label}: {UNAVAILABLE}"
    return f"{label} ({reference:.2f}): {_pct_diff(value, reference):+.1f}%"


def _trend_gap(prices: pd.DataFrame) -> str:
    label = f"{SMA_SHORT_WINDOW}-day SMA vs {SMA_LONG_WINDOW}-day SMA"
    now = _gap(prices, 0)
    if now is None:
        return f"{label}: {UNAVAILABLE}"
    then = _gap(prices, TREND_LOOKBACK_DAYS)
    if then is None:
        return f"{label}: {now:+.1f}% now; the value {TREND_LOOKBACK_DAYS} trading days ago is {UNAVAILABLE}"
    if (now > 0) != (then > 0):
        change = "averages crossed"
    else:
        change = "gap widening" if abs(now) > abs(then) else "gap narrowing" if abs(now) < abs(then) else "gap unchanged"
    return f"{label}: {now:+.1f}% now, {then:+.1f}% {TREND_LOOKBACK_DAYS} trading days ago ({change})"


def _rsi(prices: pd.DataFrame) -> str:
    label = f"RSI-{RSI_PERIOD}"
    now, then = _at(prices, COL_RSI), _at(prices, COL_RSI, RSI_LOOKBACK_DAYS)
    if now is None:
        return f"{label}: {UNAVAILABLE}"
    if then is None:
        return f"{label}: {now:.1f} now; the value {RSI_LOOKBACK_DAYS} trading days ago is {UNAVAILABLE}"
    return (
        f"{label}: {now:.1f} now, {then:.1f} {RSI_LOOKBACK_DAYS} trading days ago ({_direction(now, then)}); "
        f"overbought above {RSI_OVERBOUGHT:.0f}, oversold below {RSI_OVERSOLD:.0f}"
    )


def _macd(prices: pd.DataFrame) -> str:
    macd, signal = _at(prices, COL_MACD), _at(prices, COL_MACD_SIGNAL)
    if macd is None or signal is None:
        return f"MACD: {UNAVAILABLE}"
    side = "above" if macd > signal else "below" if macd < signal else "level with"
    text = f"MACD: {macd:.2f}, signal line {signal:.2f} (MACD {side} the signal line)"
    hist, hist_before = _at(prices, COL_MACD_HIST), _at(prices, COL_MACD_HIST, 1)
    if hist is not None and hist_before is not None:
        text += f"; histogram {hist:.2f} now, {hist_before:.2f} the day before ({_direction(hist, hist_before)})"
    return text


def _bollinger(prices: pd.DataFrame) -> str:
    return _labelled(
        "Bollinger %B",
        _at(prices, COL_BB_PCT_B),
        lambda v: f"{v:.2f} (0 is the lower band, 1 the upper band; outside 0 to 1 is outside the bands)",
    )


def _range_position(close: float | None, low: float | None, high: float | None) -> str:
    if close is None or low is None or high is None or high <= low:
        return f"52-week range: {UNAVAILABLE}"
    position = (close - low) / (high - low) * PERCENT
    return f"52-week range: low {low:.2f}, high {high:.2f}; close at {position:.0f}% of the range"


# --- helpers ----------------------------------------------------------------


def _at(prices: pd.DataFrame, column: str, days_back: int = 0) -> float | None:
    """The column's value `days_back` rows before the last, or None if missing or not finite."""
    if column not in prices.columns or len(prices) <= days_back:
        return None
    value = prices[column].iloc[-1 - days_back]
    return float(value) if pd.notna(value) and math.isfinite(value) else None


def _gap(prices: pd.DataFrame, days_back: int) -> float | None:
    short, long = _at(prices, COL_SMA_SHORT, days_back), _at(prices, COL_SMA_LONG, days_back)
    return _pct_diff(short, long) if short is not None and long is not None else None


def _pct_diff(value: float, reference: float) -> float:
    return (value / reference - 1) * PERCENT


def _direction(now: float, before: float) -> str:
    return "rising" if now > before else "falling" if now < before else "flat"


def _labelled(label: str, value: float | None, describe) -> str:
    return f"{label}: {describe(value)}" if value is not None else f"{label}: {UNAVAILABLE}"
