"""Offline end-to-end tests for task1_financial/pipeline.py, with yfinance and the news feeds replaced by fakes."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1A yfinance data pipeline and summary dictionary', Date: 2026-10-06
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1A step 2, news headlines, as designed in the grilling rounds', Date: 2026-10-06
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from task1_financial import data, news
from task1_financial.indicators import COL_SMA_LONG
from task1_financial.news import NEWS_TARGET
from task1_financial.pipeline import run_analysis, run_market_data
from task1_financial.recommendation import FALLBACK_RECOMMENDATION
from task1_financial.schemas import HeadlineReason, Recommendation
from tests.fakes import ScriptedJev, ScriptedLLM

TODAY = date(2026, 10, 6)


class FakeTicker:
    def __init__(self, ticker):
        self.info = {"longName": "Apple Inc.", "currency": "USD", "trailingPE": 38.2, "trailingEps": 8.72}


def two_years_of_bars(**_) -> pd.DataFrame:
    days = pd.bdate_range("2024-10-07", "2026-10-06", name="Date")
    close = 200.0 + np.arange(len(days)) * 0.25  # a steady uptrend
    return pd.DataFrame({"Close": close, "High": close + 1, "Low": close - 1, "Open": close, "Volume": 1_000}, index=days)


def twenty_headlines_rss(url: str) -> bytes:
    """Stands in for news._download: the same 20 recent items whichever feed is asked."""
    items = "".join(f"<item><title>Story {i}</title><pubDate>Mon, 05 Oct 2026 {i:02d}:00:00 GMT</pubDate></item>" for i in range(20))
    return f"<rss><channel>{items}</channel></rss>".encode()


def test_full_run_fills_every_field(monkeypatch):
    monkeypatch.setattr(data.yf, "download", lambda ticker, **kwargs: two_years_of_bars())
    monkeypatch.setattr(data.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(news, "_download", twenty_headlines_rss)

    result = run_market_data(" aapl ", today=TODAY)

    summary = result.summary
    assert summary["ticker"] == "AAPL"
    assert summary["as_of"] == "2026-10-06"
    for field in ["current_price", "high_52w", "low_52w", "pe_ratio", "ytd_return_pct"]:
        assert summary[field] is not None, field
    assert summary["indicators"][COL_SMA_LONG] is not None
    # Rising every day: the trend and RSI votes are up. On a straight line the MACD
    # votes come down to floating-point noise, so only a bullish score is asserted.
    assert summary["momentum"]["score"] >= 3
    assert summary["warnings"] == []
    assert COL_SMA_LONG in result.prices.columns
    assert len(result.headlines) == NEWS_TARGET
    assert result.headlines[0].title == "Story 19"  # newest first
    json.dumps(summary)


def test_yahoo_down_still_returns_a_summary(monkeypatch):
    monkeypatch.setattr(data.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(data.yf, "download", lambda ticker, **kwargs: pd.DataFrame())

    def broken_ticker(ticker):
        raise ConnectionError("no network")

    monkeypatch.setattr(data.yf, "Ticker", broken_ticker)
    monkeypatch.setattr(news, "_download", broken_ticker)

    result = run_market_data("AAPL", today=TODAY)

    summary = result.summary
    assert summary["current_price"] is None and summary["pe_ratio"] is None
    assert summary["momentum"]["label"] == "Neutral"
    assert any("price history" in w for w in summary["warnings"])
    assert any("fundamentals" in w for w in summary["warnings"])
    assert any("only 0 headlines" in w for w in summary["warnings"])
    assert result.prices.empty and result.headlines == []


def market_data(monkeypatch):
    monkeypatch.setattr(data.yf, "download", lambda ticker, **kwargs: two_years_of_bars())
    monkeypatch.setattr(data.yf, "Ticker", FakeTicker)
    monkeypatch.setattr(news, "_download", twenty_headlines_rss)
    return run_market_data("AAPL", today=TODAY)


def test_analysis_scores_every_headline_then_recommends(monkeypatch):
    market = market_data(monkeypatch)
    positive = ("positive", {"positive": 0.7, "neutral": 0.3, "negative": 0.0})
    reason = HeadlineReason(brief_reason="Growth news supports the share price.")
    call = Recommendation(
        recommendation="Buy",
        justification="The trend is up on both averages. Momentum confirms it. News adds mild support.",
        key_factors=["Close above both averages", "Positive news with rising momentum"],
    )
    jev = ScriptedJev([positive] * NEWS_TARGET)
    llm = ScriptedLLM([reason] * NEWS_TARGET + [call])

    analysis = run_analysis(market, llm=llm, jev=jev)

    assert len(analysis.headline_sentiments) == NEWS_TARGET
    assert analysis.sentiment.score == 0.7 and analysis.sentiment.label == "positive"
    assert analysis.recommendation.ok and analysis.recommendation.value.recommendation == "Buy"
    assert analysis.warnings == []
    # One Jev request and one reason per headline, then one Recommendation.
    assert len(jev.requests) == NEWS_TARGET
    assert [r["prompt"] for r in llm.requests] == ["headline_reason"] * NEWS_TARGET + ["recommendation"]


def test_analysis_finishes_when_every_model_fails(monkeypatch):
    market = market_data(monkeypatch)

    analysis = run_analysis(market, llm=ScriptedLLM([None] * (NEWS_TARGET + 1)), jev=ScriptedJev([None] * NEWS_TARGET))

    assert analysis.sentiment.score is None and analysis.sentiment.failed == NEWS_TARGET
    assert analysis.recommendation.value == FALLBACK_RECOMMENDATION and not analysis.recommendation.ok
    assert any("no headline could be scored" in w for w in analysis.warnings)
    assert any("placeholder Hold" in w for w in analysis.warnings)
