"""Task 3B's typed handoffs: Agent A's data brief, Agent B's clarification request, and A's answer.

- DataBrief (A -> B). Code copies every number from A's tool results
  (snapshots) and computes the hedge levels. A's LLM writes only the 3-5
  key observations, in one StructuredLLM call; a template writes them if the
  call fails.
- ClarificationRequest (B -> A). One StructuredLLM call over the brief and
  B's own research picks the question. `needs` can only be price_data,
  volatility or sentiment, so B can't ask for something A can't supply. If
  the call fails, DEFAULT_REQUEST asks for volatility over the 90-day
  horizon, so the critique loop still runs.
- ClarificationResponse (A -> B). A answers in its own loop with its own
  tools. Code then copies the requested data from A's latest matching
  result. If A never fetched it, the pipeline calls the tool once on A's
  behalf, traced under A with a `why` saying so, and records
  fulfilled_by="pipeline". B always gets the data it asked for.

Every handoff is a Pydantic model; the agents never pass raw text to each other.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 3: the 3B two-agent pipeline with the critique loop and the persistent cache, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import date

from common.llm import StructuredLLM
from task3_agentic.agent import make_observation
from task3_agentic.prompts import CLARIFICATION_REQUEST, KEY_OBSERVATIONS
from task3_agentic.report import (
    Snapshots,
    data_gaps,
    format_observations,
    hedge_levels,
    latest_ok,
    snapshots,
    tools_with_status_ok,
)
from task3_agentic.schemas import (
    ClarificationRequest,
    ClarificationResponse,
    DataBrief,
    KeyObservationsAnswer,
    PriceData,
    PriceSnapshot,
    SentimentSnapshot,
    SentimentStats,
    VolatilityStats,
)
from task3_agentic.tools import (
    CALCULATE_VOLATILITY,
    GET_PRICE_DATA,
    LLM_SENTIMENT,
    ToolSession,
)

logger = logging.getLogger(__name__)

ANALYST_TOOLS = (GET_PRICE_DATA, CALCULATE_VOLATILITY, LLM_SENTIMENT)
NEED_TOOLS = {"price_data": GET_PRICE_DATA, "volatility": CALCULATE_VOLATILITY, "sentiment": LLM_SENTIMENT}
HANDOFF_EFFORT = "low"  # short summaries and one choice of question; reasoning tokens count against 8K a minute
HORIZON_TRADING_DAYS = 63  # 90 calendar days

DEFAULT_REQUEST = ClarificationRequest(
    question=f"What is the annualised volatility over {HORIZON_TRADING_DAYS} trading days, the length of the 90-day horizon?",
    needs="volatility",
    period=None,
    window=HORIZON_TRADING_DAYS,
    headlines=None,
    why="The brief's 30-day volatility is shorter than the report's 90-day horizon; the longer window shows whether the expected move is understated.",
)
PIPELINE_WHY = "pipeline fallback: Agent A did not fetch the requested data, so the pipeline called the tool on A's behalf"


def build_brief(
    ticker: str,
    today: date,
    observations: Sequence[dict],
    headlines_given: int,
    llm: StructuredLLM | None,
) -> tuple[DataBrief, list[str]]:
    """Agent A's data brief from its observations, and any warnings. Never raises for a model failure."""
    warnings: list[str] = []
    found = snapshots(observations)
    fallback = KeyObservationsAnswer(key_observations=template_observations(found))
    generated_by = "template"
    if llm is None:
        _warn(warnings, "no LLM for the key observations, so a template wrote them")
        answer = fallback
    else:
        result = llm.call(
            KEY_OBSERVATIONS,
            {"ticker": ticker, "today": today.isoformat(), "observations": format_observations(observations)},
            KeyObservationsAnswer,
            fallback=fallback,
            reasoning_effort=HANDOFF_EFFORT,
        )
        if not result.ok:
            _warn(warnings, f"the key observations call failed ({result.error}), so a template wrote them")
        answer, generated_by = result.value, "llm" if result.ok else "template"

    brief = DataBrief(
        ticker=ticker,
        as_of=today,
        headlines_given=headlines_given,
        price=found.price,
        volatility=found.volatility,
        sentiment=found.sentiment,
        hedge_levels=hedge_levels(found.price, found.volatility),
        key_observations=answer.key_observations,
        data_gaps=data_gaps(observations, ANALYST_TOOLS),
        tools_called=tools_with_status_ok(observations),
        generated_by=generated_by,
    )
    return brief, warnings


def template_observations(found: Snapshots) -> list[str]:
    """Three plain observations quoting the numbers, used when A's LLM is unavailable."""
    price, vol, sentiment = found.price, found.volatility, found.sentiment
    lines = [
        (
            f"The close of {price.close:.2f} is {_signed(price.close_vs_sma_200_pct)} from the 200-day average and "
            f"{_signed(price.close_vs_sma_50_pct)} from the 50-day average."
        )
        if price
        else "No price data was available, so the trend is unknown.",
        f"{vol.window}-day volatility is {vol.current_pct:.1f}%, at the {vol.percentile_1y:.0f}th percentile of the past year."
        if vol
        else "No volatility figure was available.",
        f"The Sentiment score of the given headlines is {sentiment.score:+.2f} ({sentiment.label})."
        if sentiment and sentiment.score is not None
        else "No headline sentiment was available.",
    ]
    return lines


def write_request(brief: DataBrief, writer_observations: Sequence[dict], llm: StructuredLLM | None) -> tuple[ClarificationRequest, list[str]]:
    """Agent B's one clarification request, and any warnings. Falls back to DEFAULT_REQUEST."""
    warnings: list[str] = []
    if llm is None:
        _warn(warnings, "no LLM for the clarification request, so the default request was sent")
        return DEFAULT_REQUEST, warnings
    result = llm.call(
        CLARIFICATION_REQUEST,
        {"ticker": brief.ticker, "brief": brief.model_dump_json(), "observations": format_observations(writer_observations)},
        ClarificationRequest,
        fallback=DEFAULT_REQUEST,
        reasoning_effort=HANDOFF_EFFORT,
    )
    if not result.ok:
        _warn(warnings, f"the clarification request call failed ({result.error}), so the default request was sent")
    request = result.value
    if request.needs == "sentiment" and not request.headlines:
        _warn(warnings, "the request asked for sentiment without headlines, so the default request was sent instead")
        request = DEFAULT_REQUEST
    return request, warnings


def build_response(
    request: ClarificationRequest,
    ticker: str,
    observations: Sequence[dict],
    answer_text: str,
    session: ToolSession,
    agent: str,
) -> tuple[ClarificationResponse, list[dict], list[str]]:
    """A's typed answer, the observations behind it (including a pipeline fallback call), and warnings."""
    warnings: list[str] = []
    observations = list(observations)
    tool = NEED_TOOLS[request.needs]
    fulfilled_by = "agent"
    if latest_ok(observations, tool) is None:
        args = _requested_args(request, ticker)
        result = session.call(tool, args, agent=agent, available=ANALYST_TOOLS, why=PIPELINE_WHY)
        observations.append(make_observation(tool, args, PIPELINE_WHY, result))
        fulfilled_by = "pipeline"
        _warn(warnings, f"Agent A did not fetch the requested {request.needs}, so the pipeline called {tool} for it ({result.status})")

    obs = latest_ok(observations, tool)
    data = obs["result"]["data"] if obs else None
    if data is None:
        _warn(warnings, f"the requested {request.needs} could not be fetched; the answer has no data")
    response = ClarificationResponse(
        request=request,
        answer=answer_text.strip() or f"Fetched the requested {request.needs} with {tool}.",
        price=PriceSnapshot.from_price_data(PriceData.model_validate(data)) if data and tool == GET_PRICE_DATA else None,
        volatility=VolatilityStats.model_validate(data) if data and tool == CALCULATE_VOLATILITY else None,
        sentiment=SentimentSnapshot.from_stats(SentimentStats.model_validate(data)) if data and tool == LLM_SENTIMENT else None,
        tools_called=tools_with_status_ok(observations),
        fulfilled_by=fulfilled_by,
    )
    return response, observations, warnings


def _requested_args(request: ClarificationRequest, ticker: str) -> dict:
    """The tool arguments the request implies; the tools clamp or replace anything out of range."""
    if request.needs == "price_data":
        return {"ticker": ticker, "period": request.period or "1y"}
    if request.needs == "volatility":
        return {"ticker": ticker, "window": request.window or HORIZON_TRADING_DAYS}
    return {"headlines": request.headlines or []}


def _signed(pct: float | None) -> str:
    return "an unknown distance" if pct is None else f"{pct:+.1f}%"


def _warn(warnings: list[str], message: str) -> None:
    logger.warning(message)
    warnings.append(message)
