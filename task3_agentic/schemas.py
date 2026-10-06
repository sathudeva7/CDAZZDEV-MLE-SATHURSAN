"""Pydantic schemas for Task 3's tools: the result envelope, each tool's data, and the sentiment answer.

Every tool returns a ToolResult wrapping its own data model. The model keeps
everything (for example every price bar in the period); `digest()` is the
compact view the agent reads, because Groq's free tier allows 8K tokens a
minute and every tool message is re-sent on each agent turn.

The schemas an LLM answers (HeadlineLabels and ReportAnswer, sent as strict
JSON schemas by common/llm.py) have every field required and described.
ResearchReport is the record built from ReportAnswer: code checks the
evidence and fills in every hedge level, so the LLM never supplies a number.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 1: the five agent tools, their tests and the new-tool skill, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import json
from datetime import date
from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel, Field, field_validator

from task1_financial.news import Headline
from task1_financial.schemas import (
    HeadlineSentiment,
    Sentiment,
    SentimentSummary,
    count_sentences,
)

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


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07
# --- the research report: what the LLM answers, and the record built from it -------

ToolName = Literal["get_price_data", "calculate_volatility", "get_news", "llm_sentiment", "web_search"]
HedgeStrategyName = Literal["protective_put", "collar", "put_spread", "trim_and_stop"]
ReportSource = Literal["llm", "template"]

MIN_SUMMARY_SENTENCES = 3
MAX_SUMMARY_SENTENCES = 5
TOP_RISKS = 3
MIN_EVIDENCE = 1
MAX_EVIDENCE = 3
MAX_HEDGE_LEGS = 3
PREMIUM_NOTE = (
    "Option premiums are not priced: no tool returns option chains. Levels are indicative strikes "
    "from the expected move and price levels; price the legs from a live chain before acting."
)


class HedgeLevel(BaseModel):
    """One candidate strike or stop level, computed by code from the tool results."""

    name: str = Field(description="The name the report uses for this level, such as one_sd_low")
    level: float
    pct_from_price: float = Field(description="Distance from the current price in percent; negative is below")
    meaning: str


class HedgeLevels(BaseModel):
    """The numbers a hedge is built from: the expected 90-day move and the candidate levels around it."""

    price: float
    volatility_pct: float = Field(description="Annualised volatility used, in percent")
    volatility_source: str = Field(description="Which tool result the volatility came from")
    horizon_trading_days: int
    expected_move: float = Field(description="price x volatility x sqrt(horizon / 252): a one-standard-deviation move")
    expected_move_pct: float
    candidates: list[HedgeLevel]

    def by_name(self) -> dict[str, HedgeLevel]:
        return {candidate.name: candidate for candidate in self.candidates}


class EvidenceAnswer(BaseModel):
    fact: str = Field(description="A specific number, label or quote taken from the observations")
    source_tool: ToolName = Field(description="The tool whose result contains this fact")


class RiskAnswer(BaseModel):
    title: str = Field(description="A short name for the risk, at most 8 words")
    explanation: str = Field(description="One or two sentences on how this could move the share price in the next 90 days")
    evidence: list[EvidenceAnswer] = Field(min_length=MIN_EVIDENCE, max_length=MAX_EVIDENCE, description="One to three facts supporting the risk")


class HedgeLegAnswer(BaseModel):
    action: Literal["buy", "sell"]
    instrument: Literal["put", "call", "shares", "stop_loss"]
    level: str | None = Field(description="The name of one candidate level (such as one_sd_low); null for a shares leg")


class HedgeAnswer(BaseModel):
    strategy: HedgeStrategyName
    legs: list[HedgeLegAnswer] = Field(min_length=1, max_length=MAX_HEDGE_LEGS, description="The trades that make up the hedge")
    rationale: str = Field(description="Two to four sentences tying the strategy and its levels to the three risks and the volatility")


class ReportAnswer(BaseModel):
    """The LLM's research report, written only from the observations it is given."""

    financial_health_summary: str = Field(
        description="Three to five sentences on trend, momentum, volatility and sentiment, each quoting numbers from the observations"
    )
    top_risks: list[RiskAnswer] = Field(min_length=TOP_RISKS, max_length=TOP_RISKS, description="Exactly three distinct risks")
    hedge: HedgeAnswer

    @field_validator("financial_health_summary")
    @classmethod
    def three_to_five_sentences(cls, value: str) -> str:
        sentences = count_sentences(value)
        if not MIN_SUMMARY_SENTENCES <= sentences <= MAX_SUMMARY_SENTENCES:
            raise ValueError(
                f"financial_health_summary has {sentences} sentences; write between "
                f"{MIN_SUMMARY_SENTENCES} and {MAX_SUMMARY_SENTENCES}"
            )
        return value


class HedgeLeg(BaseModel):
    action: Literal["buy", "sell"]
    instrument: Literal["put", "call", "shares", "stop_loss"]
    level_name: str | None
    level: float | None = Field(description="Filled in by code from the candidate levels, never by the LLM")


class HedgeStrategy(BaseModel):
    strategy: HedgeStrategyName
    legs: list[HedgeLeg]
    rationale: str
    levels: HedgeLevels | None
    note: str = PREMIUM_NOTE


class Risk(BaseModel):
    title: str
    explanation: str
    evidence: list[EvidenceAnswer]


class ResearchReport(BaseModel):
    """The final research report: the LLM's sections, checked and completed by code."""

    ticker: str
    as_of: date
    financial_health_summary: str
    top_risks: list[Risk]
    hedge: HedgeStrategy
    data_gaps: list[str] = Field(description="Tools that failed or were never called, and what is missing as a result")
    tools_called: list[str] = Field(description="Every tool that returned a usable result, in call order")
    generated_by: ReportSource
    warnings: list[str] = Field(default_factory=list)
    clarification_used: str | None = Field(default=None, description="3B only: what Agent A's clarification changed")

    def to_markdown(self) -> str:
        """The report as Markdown, for display in the notebook."""
        lines = [f"# {self.ticker} research report ({self.as_of})", ""]
        if self.generated_by == "template":
            lines += ["> Written from a template: the report model was unavailable.", ""]
        lines += ["## Financial Health Summary", "", self.financial_health_summary, "", "## Top Three Risks", ""]
        for number, risk in enumerate(self.top_risks, 1):
            lines.append(f"{number}. **{risk.title}**: {risk.explanation}")
            lines += [f"   - {item.fact} _(source: {item.source_tool})_" for item in risk.evidence]
        hedge = self.hedge
        lines += ["", "## Hedge Strategy Recommendation", "", f"**{hedge.strategy.replace('_', ' ').title()}**", ""]
        for leg in hedge.legs:
            where = f" at {leg.level:.2f} ({leg.level_name})" if leg.level is not None else ""
            lines.append(f"- {leg.action} {leg.instrument.replace('_', ' ')}{where}")
        lines += ["", hedge.rationale, ""]
        if hedge.levels:
            lv = hedge.levels
            lines.append(
                f"Expected {lv.horizon_trading_days}-trading-day move: ±{lv.expected_move:.2f} "
                f"({lv.expected_move_pct:.1f}%) from {lv.price:.2f}, using {lv.volatility_pct:.1f}% volatility ({lv.volatility_source})."
            )
        lines += ["", f"_{hedge.note}_"]
        if self.clarification_used:
            lines += ["", "## What the clarification changed", "", self.clarification_used]
        if self.data_gaps:
            lines += ["", "## Data gaps", ""] + [f"- {gap}" for gap in self.data_gaps]
        return "\n".join(lines)


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 3: the 3B two-agent pipeline with the critique loop and the persistent cache, as designed in the grilling rounds', Date: 2026-10-07
# --- Task 3B: the handoff from Agent A to Agent B, and the critique loop -------------------
# Snapshots are the few numbers from each tool result that a reader needs. The DataBrief
# and the clarification answer carry these, never raw text, so the handoff is typed.

ClarificationNeed = Literal["price_data", "volatility", "sentiment"]
BriefSource = Literal["llm", "template"]
Fulfiller = Literal["agent", "pipeline"]

MIN_KEY_OBSERVATIONS = 3
MAX_KEY_OBSERVATIONS = 5


class PriceSnapshot(BaseModel):
    """The latest price figures from one get_price_data result."""

    ticker: str
    period: Period
    as_of: date
    close: float
    period_return_pct: float | None
    close_vs_sma_50_pct: float | None
    close_vs_sma_200_pct: float | None
    high_52w: float
    low_52w: float
    sma_200: float | None
    rsi_14: float | None
    macd: float | None
    macd_signal: float | None
    macd_hist: float | None
    bb_lower: float | None
    bb_pct_b: float | None
    hv_30_pct: float | None = Field(description="30-day annualised volatility from the price data, in percent")

    @classmethod
    def from_price_data(cls, price: PriceData) -> PriceSnapshot:
        latest = price.latest.indicators
        hv_30 = latest.get("hv_30")
        return cls(
            ticker=price.ticker,
            period=price.period,
            as_of=price.as_of,
            close=_round(price.latest.close),
            period_return_pct=_round(price.period_return_pct),
            close_vs_sma_50_pct=_round(price.close_vs_sma_50_pct),
            close_vs_sma_200_pct=_round(price.close_vs_sma_200_pct),
            high_52w=_round(price.high_52w),
            low_52w=_round(price.low_52w),
            sma_200=_round(latest.get("sma_200")),
            rsi_14=_round(latest.get("rsi_14")),
            macd=_round(latest.get("macd")),
            macd_signal=_round(latest.get("macd_signal")),
            macd_hist=_round(latest.get("macd_hist")),
            bb_lower=_round(latest.get("bb_lower")),
            bb_pct_b=_round(latest.get("bb_pct_b")),
            hv_30_pct=_round(hv_30 * 100) if hv_30 is not None else None,
        )


class StrongHeadline(BaseModel):
    headline: str
    sentiment: Sentiment
    score: float
    reason: str


class SentimentSnapshot(BaseModel):
    """The Sentiment score from one llm_sentiment result, with its strongest headlines."""

    score: float | None
    label: Sentiment | None
    scored: int
    counts: dict[str, int]
    strongest: list[StrongHeadline]

    @classmethod
    def from_stats(cls, stats: SentimentStats) -> SentimentSnapshot:
        digest = stats.digest()
        return cls(
            score=digest["score"], label=digest["label"], scored=digest["scored"], counts=digest["counts"],
            strongest=[StrongHeadline(**item) for item in digest["strongest"]],
        )


class KeyObservationsAnswer(BaseModel):
    """Agent A's interpretation of its own numbers, for the data brief."""

    key_observations: list[str] = Field(
        min_length=MIN_KEY_OBSERVATIONS,
        max_length=MAX_KEY_OBSERVATIONS,
        description="Three to five findings, each one sentence of at most 30 words quoting the numbers behind it",
    )


class DataBrief(BaseModel):
    """Agent A's handoff to Agent B. Code copies every number from A's tool results; A's LLM writes key_observations.

    A section is None when A never got a usable result for it; data_gaps says why.
    Sources: price from get_price_data, volatility from calculate_volatility,
    sentiment from llm_sentiment.
    """

    ticker: str
    as_of: date
    headlines_given: int = Field(description="Headlines the pipeline fetched for A before it started")
    price: PriceSnapshot | None
    volatility: VolatilityStats | None
    sentiment: SentimentSnapshot | None
    hedge_levels: HedgeLevels | None
    key_observations: list[str]
    data_gaps: list[str]
    tools_called: list[str] = Field(description="A's tools that returned a usable result")
    generated_by: BriefSource = Field(description="Who wrote key_observations: A's LLM, or a template when it failed")


class ClarificationRequest(BaseModel):
    """Agent B's one question back to Agent A. `needs` is limited to what A's tools can supply."""

    question: str = Field(description="One specific question for the quantitative analyst")
    needs: ClarificationNeed = Field(description="Which kind of data answers it: price_data, volatility or sentiment")
    period: str | None = Field(description="For price_data: 1mo, 3mo, 6mo, 1y or 2y; otherwise null")
    window: int | None = Field(description="For volatility: trading days per estimate, 5 to 252; otherwise null")
    headlines: list[str] | None = Field(description="For sentiment: headline titles you found that the brief did not score; otherwise null")
    why: str = Field(description="Which gap in the brief this fills, and how the answer will change the report")


class ClarificationResponse(BaseModel):
    """Agent A's answer to the request, with the requested data as a typed snapshot."""

    request: ClarificationRequest
    answer: str = Field(description="A's reply in its own words")
    price: PriceSnapshot | None = None
    volatility: VolatilityStats | None = None
    sentiment: SentimentSnapshot | None = None
    tools_called: list[str] = Field(description="Tools that returned a usable result while answering")
    fulfilled_by: Fulfiller = Field(description="'pipeline' when A did not fetch the data and the pipeline called the tool for it")


class FinalReportAnswer(ReportAnswer):
    """Agent B's final report: the 3A report sections plus how the clarification was used."""

    clarification_used: str = Field(description="One or two sentences on what the analyst's clarification answer changed in this report")
