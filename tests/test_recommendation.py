"""Offline tests for task1_financial/recommendation.py and the Recommendation schema.

The indicator frame is written by hand so every fact line can be checked with
a calculator. The LLM is a scripted fake (tests/fakes.py).
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from task1_financial.indicators import (
    COL_BB_PCT_B,
    COL_MACD,
    COL_MACD_HIST,
    COL_MACD_SIGNAL,
    COL_RSI,
    COL_SMA_LONG,
    COL_SMA_SHORT,
    COL_VOLATILITY,
)
from task1_financial.recommendation import (
    FALLBACK_RECOMMENDATION,
    RECOMMENDATION_HORIZON_DAYS,
    recommend,
    sentiment_facts,
    technical_facts,
)
from task1_financial.schemas import Recommendation, SentimentSummary, count_sentences
from tests.fakes import ScriptedLLM

ROWS = 12  # enough for the 10-day trend lookback plus today


def indicator_frame() -> pd.DataFrame:
    """Twelve daily rows; only today, yesterday, 5 days ago and 10 days ago matter."""
    frame = pd.DataFrame(
        {
            "Close": 105.0,
            COL_SMA_SHORT: 98.0,
            COL_SMA_LONG: 88.0,
            COL_RSI: 60.0,
            COL_MACD: 1.0,
            COL_MACD_SIGNAL: 0.5,
            COL_MACD_HIST: 0.5,
            COL_BB_PCT_B: 0.8,
            COL_VOLATILITY: 0.2,
        },
        index=pd.bdate_range("2026-09-21", periods=ROWS, name="Date"),
    )
    today, yesterday, five_ago, ten_ago = frame.index[-1], frame.index[-2], frame.index[-6], frame.index[-11]
    frame.loc[today, ["Close", COL_SMA_SHORT, COL_RSI, COL_MACD, COL_MACD_SIGNAL, COL_MACD_HIST, COL_BB_PCT_B, COL_VOLATILITY]] = [
        110.0, 100.0, 68.4, 1.5, 0.4, 1.1, 0.93, 0.241,
    ]
    frame.loc[yesterday, COL_MACD_HIST] = 0.8
    frame.loc[five_ago, COL_RSI] = 61.0
    frame.loc[ten_ago, [COL_SMA_SHORT, COL_SMA_LONG]] = [95.0, 87.5]
    return frame


SUMMARY = {
    "ticker": "AAPL",
    "company_name": "Apple Inc.",
    "currency": "USD",
    "as_of": "2026-10-06",
    "current_price": 110.0,
    "high_52w": 120.0,
    "low_52w": 80.0,
    "ytd_return_pct": 14.2,
    "momentum": {"label": "Bullish", "score": 3},
}
SENTIMENT = SentimentSummary(score=-0.0633, label="neutral", scored=14, by_jev=13, by_llm=1, failed=1, counts={"positive": 5, "negative": 4, "neutral": 5})
GOOD = Recommendation(
    recommendation="Hold",
    justification=(
        "Apple trades 10.0% above a rising 50-day average, so the uptrend is intact. "
        "Momentum is still building, with MACD above its signal line and the histogram rising. "
        "An RSI of 68.4 with %B at 0.93 shows the move is stretched near the upper band."
    ),
    key_factors=["Close above both moving averages with a widening gap: healthy uptrend", "RSI near 70 with %B 0.93: stretched"],
)


# --- sentence counting and the schema ---------------------------------------


@pytest.mark.parametrize(
    ("text", "sentences"),
    [
        ("AAPL trades 4.2% above SMA-50.", 1),  # a decimal point is not a sentence end
        ("Price rose. RSI fell.", 2),
        ("MACD sits above vs. its signal line.", 1),  # lower case after "vs." is not a new sentence
        ("Is it stretched? Yes! The bands say so.", 3),
        ("No full stop at the end", 1),
    ],
)
def test_count_sentences(text, sentences):
    assert count_sentences(text) == sentences


@pytest.mark.parametrize("sentences", [2, 6])
def test_justification_outside_three_to_five_sentences_is_rejected(sentences):
    with pytest.raises(ValidationError, match=f"has {sentences} sentences"):
        Recommendation(recommendation="Buy", justification=" ".join(["Trend and momentum agree."] * sentences), key_factors=["a", "b"])


@pytest.mark.parametrize("count", [1, 5])
def test_key_factors_need_two_to_four_items(count):
    with pytest.raises(ValidationError, match="key_factors"):
        Recommendation.model_validate(GOOD.model_dump() | {"key_factors": ["factor"] * count})


def test_fallback_is_a_valid_hold():
    assert FALLBACK_RECOMMENDATION.recommendation == "Hold"
    Recommendation.model_validate(FALLBACK_RECOMMENDATION.model_dump())


# --- facts -------------------------------------------------------------------


def test_technical_facts_are_hand_checkable():
    facts = technical_facts(indicator_frame(), SUMMARY)

    assert facts == [
        "Close: 110.00 USD",
        "Close vs 50-day SMA (100.00): +10.0%",
        "Close vs 200-day SMA (88.00): +25.0%",
        "50-day SMA vs 200-day SMA: +13.6% now, +8.6% 10 trading days ago (gap widening)",
        "RSI-14: 68.4 now, 61.0 5 trading days ago (rising); overbought above 70, oversold below 30",
        "MACD: 1.50, signal line 0.40 (MACD above the signal line); histogram 1.10 now, 0.80 the day before (rising)",
        "Bollinger %B: 0.93 (0 is the lower band, 1 the upper band; outside 0 to 1 is outside the bands)",
        "52-week range: low 80.00, high 120.00; close at 75% of the range",
        "30-day annualised volatility: 24.1%",
        "Year-to-date return: +14.2%",
    ]


def test_missing_values_are_marked_unavailable():
    frame = indicator_frame()
    frame[COL_SMA_LONG] = np.nan  # a young ticker: no 200-day average yet
    frame.loc[frame.index[-6], COL_RSI] = np.nan
    summary = SUMMARY | {"high_52w": None, "ytd_return_pct": None}

    facts = technical_facts(frame, summary)

    assert "Close vs 200-day SMA: unavailable" in facts
    assert "50-day SMA vs 200-day SMA: unavailable" in facts
    assert "RSI-14: 68.4 now; the value 5 trading days ago is unavailable" in facts
    assert "52-week range: unavailable" in facts
    assert "Year-to-date return: unavailable" in facts


def test_no_prices_gives_every_fact_unavailable():
    # With no prices, build_summary() also leaves the price-based fields empty.
    no_prices = {"current_price": None, "high_52w": None, "low_52w": None, "ytd_return_pct": None}
    facts = technical_facts(indicator_frame().iloc[0:0], SUMMARY | no_prices)

    assert all(fact.endswith("unavailable") for fact in facts)


def test_sentiment_facts():
    assert sentiment_facts(SENTIMENT) == [
        (
            "Sentiment score: -0.06 (neutral) on a scale from -1 to +1, from 14 headlines of the last 7 days "
            "(5 positive, 4 negative, 5 neutral)"
        )
    ]
    empty = SentimentSummary(score=None, label=None, scored=0, by_jev=0, by_llm=0, failed=3, counts={})
    assert sentiment_facts(empty) == ["Sentiment score: unavailable (no headline could be scored)"]


# --- the call ------------------------------------------------------------------


def test_recommend_sends_facts_without_the_momentum_label():
    llm = ScriptedLLM([GOOD])

    result = recommend(indicator_frame(), SUMMARY, SENTIMENT, llm)

    assert result.ok and result.value == GOOD and result.provider == "groq"
    [request] = llm.requests
    assert request["prompt"] == "recommendation" and request["effort"] == "medium"
    variables = request["variables"]
    assert variables["horizon_days"] == RECOMMENDATION_HORIZON_DAYS
    assert "Close vs 50-day SMA (100.00): +10.0%" in variables["technical_facts"]
    assert "Sentiment score: -0.06" in variables["sentiment_facts"]
    # Our rule-based Momentum signal is kept out, so the model cannot just copy it.
    sent = " ".join(str(value) for value in variables.values())
    assert "Bullish" not in sent and "momentum" not in sent.lower()
    assert result.facts == variables["technical_facts"].split("\n") + variables["sentiment_facts"].split("\n")


def test_recommend_falls_back_to_a_labelled_hold():
    result = recommend(indicator_frame(), SUMMARY, SENTIMENT, ScriptedLLM([None]))

    assert not result.ok and result.provider is None
    assert result.value == FALLBACK_RECOMMENDATION
