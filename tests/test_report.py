"""Offline tests for task1_financial/report.py, on hand-built pipeline outputs."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1 notebook and the bonus report as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from task1_financial.indicators import add_indicators
from task1_financial.news import Headline
from task1_financial.pipeline import Analysis, MarketData
from task1_financial.recommendation import FALLBACK_RECOMMENDATION, RecommendationResult
from task1_financial.report import (
    NO_HEADLINES,
    RISK_DISCLAIMER,
    build_report,
    plot_price_chart,
    save_report,
    top_headlines,
)
from task1_financial.schemas import HeadlineSentiment, Recommendation, SentimentSummary

GENERATED_AT = datetime(2026, 10, 6, 16, 30, tzinfo=timezone.utc)
NEWEST = datetime(2026, 10, 6, 12, 0, tzinfo=timezone.utc)
REQUIRED_SECTIONS = ["## Company snapshot", "## Technical outlook", "## News sentiment", "## Recommendation"]

RECOMMENDATION = Recommendation(
    recommendation="Buy",
    justification=(
        "Price sits above both moving averages and the gap between them is widening. "
        "RSI is rising but below overbought, so the trend has room. "
        "Positive news supports the technical picture."
    ),
    key_factors=["Price above SMA 50 and SMA 200 with a widening gap", "RSI rising below 70 with MACD above signal"],
)


def prices(days: int = 520) -> pd.DataFrame:
    index = pd.bdate_range(end="2026-10-06", periods=days, name="Date")
    close = 200.0 + np.cumsum(np.sin(np.arange(days) / 9.0))  # waves, so RSI and MACD move both ways
    bars = pd.DataFrame({"Close": close, "High": close + 1, "Low": close - 1, "Open": close, "Volume": 1_000}, index=index)
    return add_indicators(bars)


def summary(**overrides) -> dict:
    base = {
        "ticker": "AAPL",
        "company_name": "Apple Inc.",
        "currency": "USD",
        "as_of": "2026-10-06",
        "current_price": 210.5,
        "high_52w": 230.0,
        "low_52w": 180.0,
        "pe_ratio": 31.2,
        "ytd_return_pct": 4.3,
        "momentum": {"label": "Bullish", "score": 3},
        "indicators": {},
        "warnings": [],
    }
    return {**base, **overrides}


def headline(title: str, hours_old: int) -> Headline:
    return Headline(title=title, publisher="Reuters", published=NEWEST - timedelta(hours=hours_old), url=None, feed="yahoo")


def scored(title: str, score: float | None, scored_by: str = "jev") -> HeadlineSentiment:
    sentiment = "neutral" if score is None or abs(score) < 0.2 else ("positive" if score > 0 else "negative")
    return HeadlineSentiment(
        headline=title, sentiment=sentiment, confidence=0.8, brief_reason=f"Reason for {title}.",
        probabilities=None, scored_by=scored_by if score is not None else "none", score=score,
    )


def build(scores=(0.9, -0.3, 0.1, -0.9, None), rec_ok=True, warnings=(), **summary_overrides):
    """A MarketData and Analysis pair; headline i is i hours older than headline i - 1."""
    titles = [f"Story {i}" for i in range(len(scores))]
    market = MarketData(
        prices=prices(),
        summary=summary(**summary_overrides),
        headlines=[headline(title, hours_old=i) for i, title in enumerate(titles)],
    )
    items = [scored(title, score) for title, score in zip(titles, scores)]
    kept = [s for s in scores if s is not None]
    sentiment = SentimentSummary(
        score=round(sum(kept) / len(kept), 4) if kept else None,
        label="neutral" if kept else None,
        scored=len(kept), by_jev=len(kept), by_llm=0, failed=len(scores) - len(kept),
        counts={"positive": sum(s > 0.2 for s in kept), "negative": sum(s < -0.2 for s in kept), "neutral": sum(abs(s) <= 0.2 for s in kept)},
    )
    rec = RecommendationResult(
        value=RECOMMENDATION if rec_ok else FALLBACK_RECOMMENDATION,
        ok=rec_ok, provider="groq" if rec_ok else None, facts=[],
    )
    return market, Analysis(headline_sentiments=items, sentiment=sentiment, recommendation=rec, warnings=list(warnings))


def test_every_required_section_and_the_disclaimer_are_present():
    report = build_report(*build(), generated_at=GENERATED_AT)

    for section in REQUIRED_SECTIONS:
        assert section in report.markdown, section
    assert "Recommendation: Buy over the next 90 days" in report.markdown
    assert RISK_DISCLAIMER in report.markdown
    assert "Apple Inc. (AAPL): equity research brief" in report.markdown
    assert "Generated 2026-10-06 16:30 UTC" in report.markdown
    assert "210.50 USD" in report.markdown and "+4.3%" in report.markdown
    assert "openai/gpt-oss-120b via groq" in report.markdown
    assert "## Data notes" not in report.markdown  # no warnings, no section
    assert "<h2>Company snapshot</h2>" in report.html and "<table>" in report.html


def test_top_three_are_the_strongest_either_way_with_ties_to_the_newer():
    market, analysis = build(scores=(0.9, -0.3, 0.1, -0.9, None, 0.3))

    top = top_headlines(market, analysis)

    # |0.9| ties: Story 0 is newer than Story 3. |0.3| ties: Story 1 is newer than Story 5.
    assert [item.headline for item, _ in top] == ["Story 0", "Story 3", "Story 1"]
    assert all(source is not None and source.title == item.headline for item, source in top)


def test_missing_values_read_unavailable():
    report = build_report(*build(pe_ratio=None, ytd_return_pct=None, as_of=None), generated_at=GENERATED_AT)

    assert "| P/E ratio (trailing) | unavailable |" in report.markdown
    assert "| Year-to-date return | unavailable |" in report.markdown
    assert "market data as of unavailable" in report.markdown


def test_no_headlines_is_said_on_the_page():
    market, analysis = build(scores=())
    market.headlines.clear()

    report = build_report(market, analysis, generated_at=GENERATED_AT)

    assert NO_HEADLINES in report.markdown
    assert "Top 3 headlines" not in report.markdown


def test_placeholder_hold_is_flagged_and_warnings_become_data_notes():
    warning = "the Recommendation is a placeholder Hold: no LLM provider answered"
    report = build_report(*build(rec_ok=False, warnings=[warning]), generated_at=GENERATED_AT)

    assert "Recommendation: Hold" in report.markdown
    assert "**Placeholder:**" in report.markdown
    assert "## Data notes" in report.markdown and warning in report.markdown
    assert "no model (placeholder)" in report.markdown


def test_html_embeds_the_chart_and_markdown_points_at_the_png(tmp_path):
    report = build_report(*build(), generated_at=GENERATED_AT)

    assert 'src="data:image/png;base64,' in report.html
    assert "(AAPL_chart.png)" in report.markdown
    assert report.chart_png is not None and report.chart_png.startswith(b"\x89PNG")

    paths = save_report(report, tmp_path)
    assert sorted(p.name for p in paths) == ["AAPL_brief.html", "AAPL_brief.md", "AAPL_chart.png"]


def test_headline_text_cannot_inject_html_or_markdown():
    market, analysis = build(scores=(0.9,))
    analysis.headline_sentiments[0] = scored("<script>x</script> *wow*", 0.9)

    report = build_report(market, analysis, generated_at=GENERATED_AT)

    assert "<script>" not in report.html
    assert "<em>wow</em>" not in report.html


@pytest.mark.parametrize("days", [0, 30, 120])
def test_chart_draws_on_short_or_empty_history(days):
    fig = plot_price_chart(prices(days) if days else pd.DataFrame(), "AAPL")

    assert len(fig.axes) == 3
