"""Pydantic schemas for Task 3's tools: the result envelope, each tool's data, and the sentiment answer.

Every tool returns a ToolResult wrapping its own data model. The model keeps
everything (for example every price bar in the period); `digest()` is the
compact view the agent reads, because Groq's free tier allows 8K tokens a
minute and every tool message is re-sent on each agent turn.

The one schema an LLM answers is HeadlineLabels (sent as a strict JSON schema
by common/llm.py), so every field is required and described.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 1: the five agent tools, their tests and the new-tool skill, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import json
from datetime import date
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field

from task1_financial.news import Headline
from task1_financial.schemas import HeadlineSentiment, Sentiment, SentimentSummary

DIGEST_DECIMALS = 2
DIGEST_LAST_BARS = 5  # recent bars shown to the agent; the typed result keeps the whole period
DIGEST_STRONGEST_HEADLINES = 3
DIGEST_SNIPPET_CHARS = 300

Status = Literal["ok", "empty", "error"]
Period = Literal["1mo", "3mo", "6mo", "1y", "2y"]
SearchBackend = Literal["text", "news"]

DataT = TypeVar("DataT", bound=BaseModel)


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, DIGEST_DECIMALS)


# --- the envelope every tool returns ----------------------------------------


class ToolResult(BaseModel, Generic[DataT]):
    """One tool call's outcome. Tools never raise: a failure is status 'error' or 'empty' plus a hint.

    - ok: `data` holds the answer (it may still carry warnings).
    - empty: the source answered, after retries, with nothing usable (no bars, no headlines, no hits).
    - error: the tool itself failed (an LLM failure, an injected demo failure, or a bug).
    """

    tool: str
    status: Status
    data: DataT | None
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    hint: str | None = Field(default=None, description="What the agent could try instead; set when status is not ok")

    def for_llm(self) -> str:
        """The JSON text the agent reads: the data's digest, with empty fields left out to save tokens."""
        view: dict[str, Any] = {"tool": self.tool, "status": self.status}
        if self.data is not None:
            view["data"] = self.data.digest()
        for key in ("error", "warnings", "hint"):
            if getattr(self, key):
                view[key] = getattr(self, key)
        return json.dumps(view, default=str)


# --- get_price_data ----------------------------------------------------------


class PriceBar(BaseModel):
    """One trading day: adjusted OHLCV plus every Task 1 indicator, keyed by indicators.COL_* names."""

    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    indicators: dict[str, float | None]


class PriceData(BaseModel):
    """Daily bars for the requested period, with the figures an analyst reads first."""

    ticker: str
    period: Period
    as_of: date = Field(description="Date of the latest bar")
    bars: list[PriceBar] = Field(description="Every bar in the period, oldest first")
    period_return_pct: float | None = Field(description="Close-to-close change over the period, in percent")
    period_high: float
    period_low: float
    high_52w: float
    low_52w: float
    close_vs_sma_50_pct: float | None = Field(description="How far the close is above (+) or below (-) the 50-day SMA, in percent")
    close_vs_sma_200_pct: float | None

    @property
    def latest(self) -> PriceBar:
        return self.bars[-1]

    def digest(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "period": self.period,
            "as_of": self.as_of,
            "close": _round(self.latest.close),
            "period_return_pct": _round(self.period_return_pct),
            "period_high": _round(self.period_high),
            "period_low": _round(self.period_low),
            "high_52w": _round(self.high_52w),
            "low_52w": _round(self.low_52w),
            "close_vs_sma_50_pct": _round(self.close_vs_sma_50_pct),
            "close_vs_sma_200_pct": _round(self.close_vs_sma_200_pct),
            "indicators": {name: _round(value) for name, value in self.latest.indicators.items()},
            "last_bars": [
                [bar.date, _round(bar.open), _round(bar.high), _round(bar.low), _round(bar.close), int(bar.volume)]
                for bar in self.bars[-DIGEST_LAST_BARS:]
            ],
            "last_bars_columns": ["date", "open", "high", "low", "close", "volume"],
        }


# --- calculate_volatility ----------------------------------------------------


class VolatilityStats(BaseModel):
    """Annualised historical volatility now, and where it sits within the past year."""

    ticker: str
    window: int = Field(description="Trading days of log returns in each estimate")
    as_of: date
    current_pct: float = Field(description="Annualised volatility today, in percent (28.4 means 28.4%)")
    min_1y_pct: float
    median_1y_pct: float
    max_1y_pct: float
    percentile_1y: float = Field(ge=0, le=100, description="Share of the past year's daily values at or below today's")
    observations: int = Field(description="Daily volatility values in the one-year comparison")

    def digest(self) -> dict[str, Any]:
        return {key: _round(value) if isinstance(value, float) else value for key, value in self.model_dump().items()}


# --- get_news ----------------------------------------------------------------


class NewsList(BaseModel):
    """Recent headlines for a ticker, newest first (Task 1's Headline records)."""

    ticker: str
    headlines: list[Headline]

    def digest(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "count": len(self.headlines),
            "headlines": [
                {"title": h.title, "publisher": h.publisher, "published": h.published.date()} for h in self.headlines
            ],
        }


# --- llm_sentiment -----------------------------------------------------------


class HeadlineLabel(BaseModel):
    """The LLM's judgement of one numbered headline in a batch."""

    index: int = Field(ge=1, description="The headline's number in the list, starting at 1")
    sentiment: Sentiment = Field(description="Likely effect on the share price of the company the headline is about")
    confidence: float = Field(ge=0, le=1, description="How sure you are of the sentiment, from 0 to 1")
    reason: str = Field(description="At most 15 words naming the mechanism, such as earnings, demand, regulation or analyst views")


class HeadlineLabels(BaseModel):
    """The LLM's answer for a whole batch of headlines, one label per headline."""

    labels: list[HeadlineLabel] = Field(description="One entry per headline, in the order given")


class SentimentStats(BaseModel):
    """The Sentiment score for a batch of headlines, computed with Task 1's rule, and each headline's label."""

    summary: SentimentSummary
    items: list[HeadlineSentiment] = Field(description="Same order as the headlines given")

    def digest(self) -> dict[str, Any]:
        scored = [item for item in self.items if item.score is not None]
        # sorted() is stable, so equally strong headlines keep the order they were given in.
        strongest = sorted(scored, key=lambda item: abs(item.score), reverse=True)[:DIGEST_STRONGEST_HEADLINES]
        return {
            "score": self.summary.score,
            "label": self.summary.label,
            "scored": self.summary.scored,
            "failed": self.summary.failed,
            "counts": self.summary.counts,
            "strongest": [
                {"headline": item.headline, "sentiment": item.sentiment, "score": _round(item.score), "reason": item.brief_reason}
                for item in strongest
            ],
        }


# --- web_search --------------------------------------------------------------


class SearchHit(BaseModel):
    title: str
    url: str | None
    snippet: str
    published: str | None = Field(description="Publication time as the search engine gave it; news results only")


class SearchResults(BaseModel):
    """The top results for one query, and which search produced them."""

    query: str
    backend: SearchBackend = Field(description="'text' is the web search; 'news' is the fallback used when it fails")
    hits: list[SearchHit]

    def digest(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "backend": self.backend,
            "hits": [
                {"title": hit.title, "snippet": hit.snippet[:DIGEST_SNIPPET_CHARS], "url": hit.url, "published": hit.published}
                for hit in self.hits
            ],
        }
