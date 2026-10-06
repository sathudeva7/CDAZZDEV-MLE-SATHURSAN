"""The research report: hedge levels computed by code, sections written by the LLM, then checked.

1. Hedge levels. The expected move over the horizon is
       price x annualised volatility x sqrt(horizon trading days / 252)
   With price 200, volatility 30% and 63 trading days (about 90 calendar
   days): 200 x 0.30 x sqrt(63/252) = 200 x 0.30 x 0.5 = 30, so about two
   times in three the price ends between 170 and 230. The candidate levels
   are those two bounds plus the SMA-200, the lower Bollinger Band and the
   52-week low, each with its distance from the price.
2. Sections. One StructuredLLM call (RESEARCH_REPORT) writes the summary,
   three risks and the hedge choice from the agent's observations: the same
   digests the agent read. Hedge legs name a candidate level; code fills in
   the number, so the LLM never supplies a price.
3. Checks. Evidence must cite a tool that returned status ok in this run;
   anything else is dropped with a warning. An unknown level name leaves the
   leg without a level, with a warning.
4. Fallback. When no LLM answers, a template report built from the same
   observations takes its place, labelled generated_by="template".

An observation is the dict the agent loop stores for each tool call (see
agent.py): tool, args, why, status, digest (the text the agent read) and
result (the ToolResult as JSON, or None when the call never ran).
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from datetime import date

from common.llm import StructuredLLM
from task1_financial.indicators import (
    COL_BB_LOWER,
    COL_SMA_LONG,
    COL_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
)
from task3_agentic.prompts import RESEARCH_REPORT
from task3_agentic.schemas import (
    EvidenceAnswer,
    HedgeAnswer,
    HedgeLeg,
    HedgeLegAnswer,
    HedgeLevel,
    HedgeLevels,
    HedgeStrategy,
    PriceData,
    ReportAnswer,
    ResearchReport,
    Risk,
    RiskAnswer,
    SentimentStats,
    VolatilityStats,
)
from task3_agentic.tools import (
    CALCULATE_VOLATILITY,
    GET_PRICE_DATA,
    LLM_SENTIMENT,
    TOOL_NAMES,
)

logger = logging.getLogger(__name__)

HEDGE_HORIZON_TRADING_DAYS = 63  # 90 calendar days is about 63 trading days
PERCENT = 100.0
LEVEL_DECIMALS = 2
REPORT_EFFORT = "medium"  # reasons across price, volatility, sentiment and commentary at once
NO_HEDGE_LEVELS = "unavailable: no usable price data, so no levels could be computed"

# Candidate level names and what each one means, in the order the prompt lists them.
ONE_SD_LOW = "one_sd_low"
ONE_SD_HIGH = "one_sd_high"
SMA_200 = "sma_200"
BB_LOWER = "bb_lower"
LOW_52W = "low_52w"
LEVEL_MEANINGS = {
    ONE_SD_LOW: "price minus one expected move: the bottom of the likely 90-day range",
    ONE_SD_HIGH: "price plus one expected move: the top of the likely 90-day range",
    SMA_200: "the 200-day moving average, the long-term trend line",
    BB_LOWER: "the lower Bollinger Band, two standard deviations below the 20-day average",
    LOW_52W: "the lowest price of the past 52 weeks",
}


def compute_hedge_levels(observations: Sequence[dict]) -> HedgeLevels | None:
    """The expected move and candidate levels from the latest usable price and volatility results."""
    price_obs = latest_ok(observations, GET_PRICE_DATA)
    if price_obs is None:
        return None
    price = PriceData.model_validate(price_obs["result"]["data"])
    close = price.latest.close

    vol_obs = latest_ok(observations, CALCULATE_VOLATILITY)
    if vol_obs is not None:
        stats = VolatilityStats.model_validate(vol_obs["result"]["data"])
        volatility_pct, source = stats.current_pct, f"calculate_volatility, {stats.window}-day"
    elif price.latest.indicators.get(COL_VOLATILITY) is not None:
        volatility_pct, source = price.latest.indicators[COL_VOLATILITY] * PERCENT, f"get_price_data {COL_VOLATILITY}"
    else:
        return None

    move = expected_move(close, volatility_pct)
    levels = {ONE_SD_LOW: close - move, ONE_SD_HIGH: close + move}
    for name, value in ((SMA_200, price.latest.indicators.get(COL_SMA_LONG)), (BB_LOWER, price.latest.indicators.get(COL_BB_LOWER)), (LOW_52W, price.low_52w)):
        if value is not None:
            levels[name] = value
    candidates = [
        HedgeLevel(name=name, level=round(value, LEVEL_DECIMALS), pct_from_price=round((value / close - 1) * PERCENT, LEVEL_DECIMALS), meaning=LEVEL_MEANINGS[name])
        for name, value in levels.items()
    ]
    return HedgeLevels(
        price=round(close, LEVEL_DECIMALS),
        volatility_pct=round(volatility_pct, LEVEL_DECIMALS),
        volatility_source=source,
        horizon_trading_days=HEDGE_HORIZON_TRADING_DAYS,
        expected_move=round(move, LEVEL_DECIMALS),
        expected_move_pct=round(move / close * PERCENT, LEVEL_DECIMALS),
        candidates=candidates,
    )


def expected_move(price: float, volatility_pct: float, trading_days: int = HEDGE_HORIZON_TRADING_DAYS) -> float:
    """One standard deviation of price change over `trading_days`: 200, 30%, 63 days gives 30.0."""
    return price * volatility_pct / PERCENT * math.sqrt(trading_days / TRADING_DAYS_PER_YEAR)


def write_report(
    ticker: str,
    today: date,
    observations: Sequence[dict],
    llm: StructuredLLM | None,
    tools_available: Sequence[str] = TOOL_NAMES,
) -> ResearchReport:
    """The checked research report. Never raises for a model failure: the template takes over."""
    warnings: list[str] = []
    levels = compute_hedge_levels(observations)
    fallback = template_answer(observations, levels)
    if llm is None:
        _warn(warnings, "no LLM is available for the report, so it was written from a template")
        answer, generated_by = fallback, "template"
    else:
        result = llm.call(
            RESEARCH_REPORT,
            {
                "ticker": ticker,
                "today": today.isoformat(),
                "observations": format_observations(observations),
                "hedge_levels": levels.model_dump_json() if levels else NO_HEDGE_LEVELS,
            },
            ReportAnswer,
            fallback=fallback,
            reasoning_effort=REPORT_EFFORT,
        )
        if not result.ok:
            _warn(warnings, f"the report model failed ({result.error}), so the report was written from a template")
        answer, generated_by = result.value, "llm" if result.ok else "template"

    ok_tools = tools_with_status_ok(observations)
    return ResearchReport(
        ticker=ticker,
        as_of=today,
        financial_health_summary=answer.financial_health_summary,
        top_risks=[_checked_risk(risk, ok_tools, warnings) for risk in answer.top_risks],
        hedge=_hedge_strategy(answer.hedge, levels, warnings),
        data_gaps=data_gaps(observations, tools_available),
        tools_called=ok_tools,
        generated_by=generated_by,
        warnings=warnings,
    )


def format_observations(observations: Sequence[dict]) -> str:
    """The observations as the report prompt sees them: each call's arguments and the digest the agent read."""
    rows = []
    for number, obs in enumerate(observations, 1):
        digest = json.loads(obs["digest"]) if obs.get("digest") else {"tool": obs["tool"], "status": obs["status"]}
        rows.append({"call": number, "args": obs.get("args"), **digest})
    return json.dumps(rows, default=str)


def latest_ok(observations: Sequence[dict], tool: str) -> dict | None:
    """The most recent observation of `tool` that returned status ok."""
    for obs in reversed(observations):
        if obs["tool"] == tool and obs["status"] == "ok" and obs.get("result"):
            return obs
    return None


def tools_with_status_ok(observations: Sequence[dict]) -> list[str]:
    """Each tool that returned at least one ok result, in first-call order."""
    names: list[str] = []
    for obs in observations:
        if obs["status"] == "ok" and obs["tool"] not in names:
            names.append(obs["tool"])
    return names


def data_gaps(observations: Sequence[dict], tools_available: Sequence[str]) -> list[str]:
    """One line per available tool that was never called or never succeeded."""
    gaps = []
    for tool in tools_available:
        calls = [obs for obs in observations if obs["tool"] == tool]
        if not calls:
            gaps.append(f"{tool} was not called")
        elif not any(obs["status"] == "ok" for obs in calls):
            last = calls[-1]
            gaps.append(f"{tool} returned no usable result ({last['status']}: {_problem(last)})")
    return gaps


def template_answer(observations: Sequence[dict], levels: HedgeLevels | None) -> ReportAnswer:
    """A plain report from the observations alone, used when no LLM can write one.

    It quotes the numbers directly, so it is accurate but generic: the three
    risks are always trend, volatility and news sentiment.
    """
    price_obs = latest_ok(observations, GET_PRICE_DATA)
    vol_obs = latest_ok(observations, CALCULATE_VOLATILITY)
    sent_obs = latest_ok(observations, LLM_SENTIMENT)
    price = PriceData.model_validate(price_obs["result"]["data"]) if price_obs else None
    vol = VolatilityStats.model_validate(vol_obs["result"]["data"]) if vol_obs else None
    sentiment = SentimentStats.model_validate(sent_obs["result"]["data"]).summary if sent_obs else None

    if price:
        trend = (
            f"The price closed at {price.latest.close:.2f} on {price.as_of}, "
            f"{_signed(price.close_vs_sma_200_pct)} from its 200-day average and {_signed(price.close_vs_sma_50_pct)} from its 50-day average."
        )
        trend_fact = EvidenceAnswer(fact=f"Close {_signed(price.close_vs_sma_200_pct)} vs the 200-day average", source_tool="get_price_data")
    else:
        trend = "Price data was unavailable, so the trend could not be assessed."
        trend_fact = EvidenceAnswer(fact="No price data was returned", source_tool="get_price_data")
    if vol:
        volatility = f"Annualised {vol.window}-day volatility is {vol.current_pct:.1f}%, at the {vol.percentile_1y:.0f}th percentile of the past year."
        vol_fact = EvidenceAnswer(fact=f"Volatility {vol.current_pct:.1f}%, percentile {vol.percentile_1y:.0f}", source_tool="calculate_volatility")
    else:
        volatility = "Volatility was unavailable."
        vol_fact = EvidenceAnswer(fact="No volatility result was returned", source_tool="calculate_volatility")
    if sentiment and sentiment.score is not None:
        news = f"The Sentiment score of recent headlines is {sentiment.score:+.2f} ({sentiment.label})."
        news_fact = EvidenceAnswer(fact=f"Sentiment score {sentiment.score:+.2f} over {sentiment.scored} headlines", source_tool="llm_sentiment")
    else:
        news = "Headline sentiment was unavailable."
        news_fact = EvidenceAnswer(fact="No sentiment result was returned", source_tool="llm_sentiment")

    if levels:
        hedge = HedgeAnswer(
            strategy="protective_put",
            legs=[HedgeLegAnswer(action="buy", instrument="put", level=ONE_SD_LOW)],
            rationale=(
                "A put at the bottom of the likely 90-day range limits losses from a move larger than usual. "
                "It is the simplest hedge that covers all three risks at once."
            ),
        )
    else:
        hedge = HedgeAnswer(
            strategy="trim_and_stop",
            legs=[HedgeLegAnswer(action="sell", instrument="shares", level=None)],
            rationale="Without price data no option levels can be set, so reducing the position is the only hedge that can be sized.",
        )
    return ReportAnswer(
        financial_health_summary=f"{trend} {volatility} {news}",
        top_risks=[
            RiskAnswer(title="Trend reversal", explanation="A break in the price trend could extend losses over the next 90 days.", evidence=[trend_fact]),
            RiskAnswer(title="Volatility", explanation="Larger than usual daily moves widen the range of likely 90-day outcomes.", evidence=[vol_fact]),
            RiskAnswer(title="News sentiment", explanation="Negative news flow can weigh on the share price.", evidence=[news_fact]),
        ],
        hedge=hedge,
    )


def _checked_risk(risk: RiskAnswer, ok_tools: Sequence[str], warnings: list[str]) -> Risk:
    """The risk with only the evidence that cites a tool which returned an ok result in this run."""
    kept = [item for item in risk.evidence if item.source_tool in ok_tools]
    for item in risk.evidence:
        if item.source_tool not in ok_tools:
            _warn(warnings, f"dropped evidence for '{risk.title}' citing {item.source_tool}, which returned no usable result: {item.fact!r}")
    if not kept:
        _warn(warnings, f"risk '{risk.title}' has no supporting evidence left")
    return Risk(title=risk.title, explanation=risk.explanation, evidence=kept)


def _hedge_strategy(answer: HedgeAnswer, levels: HedgeLevels | None, warnings: list[str]) -> HedgeStrategy:
    named = levels.by_name() if levels else {}
    legs = []
    for leg in answer.legs:
        level = named.get(leg.level) if leg.level else None
        if leg.level and level is None:
            _warn(warnings, f"hedge leg '{leg.action} {leg.instrument}' names an unknown level {leg.level!r}, so it has no price")
        legs.append(HedgeLeg(action=leg.action, instrument=leg.instrument, level_name=leg.level, level=level.level if level else None))
    return HedgeStrategy(strategy=answer.strategy, legs=legs, rationale=answer.rationale, levels=levels)


def _problem(obs: dict) -> str:
    result = obs.get("result") or {}
    return result.get("error") or "; ".join(result.get("warnings") or []) or "no detail"


def _signed(pct: float | None) -> str:
    return "an unknown distance" if pct is None else f"{pct:+.1f}%"


def _warn(warnings: list[str], message: str) -> None:
    """Log a check or fallback and record it on the report, so the two always agree."""
    logger.warning(message)
    warnings.append(message)
