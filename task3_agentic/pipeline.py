"""Task 3B: the two-agent research pipeline, with the critique loop and the persistent cache.

    START -> cache_lookup --(hit)--> END
                 |
               (miss)
                 v
    news_intake -> analyst (A) -> writer (B) -> analyst_answer (A) -> final_report (B) -> cache_save -> END
                       |              |                 |
                   DataBrief   ClarificationRequest   ClarificationResponse

- news_intake: the pipeline calls get_news once, because Agent A has no news
  tool and its brief needs headline sentiment. The call is traced under
  agent "pipeline".
- analyst: Agent A's loop with get_price_data, calculate_volatility and
  llm_sentiment only. Its finish step builds the DataBrief (handoff.py).
- writer: Agent B's loop with web_search and get_news only, given the brief.
  Its finish step writes the one ClarificationRequest.
- analyst_answer: Agent A again, in a fresh thread, with its own tools, to
  answer the request. Its finish step builds the ClarificationResponse.
- final_report: B's report from the brief, its research and the response.

The critique is a fixed edge, so it runs exactly once per pipeline run. What
B asks, and which tools each agent calls, is up to the models. Each agent's
tools are bound in build_agent_graph, so restriction does not depend on
prompts. Every node reports to `on_update(agent, step)` for the notebook trace.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 3: the 3B two-agent pipeline with the critique loop and the persistent cache, as designed in the grilling rounds', Date: 2026-10-07
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add a stronger paid OpenAI model (gpt-6.1-sol) for testing, in a separate file so it can be deleted before submission', Date: 2026-10-07

from __future__ import annotations

import logging
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from common.llm import StructuredLLM
from common.llm_config import active_profile
from task3_agentic.agent import (
    AgentState,
    build_agent_graph,
    chat_models,
    invoke_agent,
    message_text,
)
from task3_agentic.cache import (
    CACHE_DIR,
    SCHEMA_VERSION,
    CachedRun,
    cache_path,
    load_cached,
    save_cached,
)
from task3_agentic.handoff import (
    ANALYST_TOOLS,
    build_brief,
    build_response,
    write_request,
)
from task3_agentic.prompts import CLARIFICATION_TASK, DATA_ANALYST, RESEARCH_WRITER
from task3_agentic.report import write_final_report
from task3_agentic.schemas import (
    ClarificationRequest,
    ClarificationResponse,
    DataBrief,
    ResearchReport,
)
from task3_agentic.tools import GET_NEWS, WEB_SEARCH, ToolSession

logger = logging.getLogger(__name__)

ANALYST, WRITER, PIPELINE = "A", "B", "pipeline"
WRITER_TOOLS = (WEB_SEARCH, GET_NEWS)
INTAKE_HEADLINES = 10
# Smaller than 3A's budget of 8: each agent covers only half the job, and the
# whole pipeline has to fit Groq's free 8K tokens a minute.
ANALYST_MAX_TOOL_CALLS = 6
WRITER_MAX_TOOL_CALLS = 5
ANSWER_MAX_TOOL_CALLS = 3


class PipelineState(TypedDict):
    ticker: str
    force_refresh: bool
    headlines: list[str]
    brief: dict | None
    writer_observations: list[dict]
    request: dict | None
    response: dict | None
    report: dict | None
    cached: bool
    warnings: Annotated[list[str], operator.add]


@dataclass
class PipelineRun:
    """One pipeline run: the report, the handoffs behind it, and whether it came from the cache."""

    run_id: str
    report: ResearchReport
    brief: DataBrief
    request: ClarificationRequest
    response: ClarificationResponse
    cached: bool
    cache_path: Path
    warnings: list[str] = field(default_factory=list)


class ResearchPipeline:
    """Agent A (data analyst) and Agent B (research writer), coordinated by a fixed graph."""

    def __init__(
        self,
        session: ToolSession,
        *,
        models: Sequence[BaseChatModel] | None = None,
        llm: StructuredLLM | None = None,
        cache_dir: Path | str = CACHE_DIR,
    ) -> None:
        self.session = session
        self.cache_dir = Path(cache_dir)
        self._llm = llm
        self._on_update: Callable | None = None
        models = models or chat_models()
        checkpointer = InMemorySaver()

        def agent(name, tools, prompt, finish, budget):
            return build_agent_graph(
                session=session, tool_names=tools, agent_name=name, models=models, system_prompt=prompt.system,
                finish=finish, checkpointer=checkpointer, max_tool_calls=budget,
            )

        self.analyst = agent(ANALYST, ANALYST_TOOLS, DATA_ANALYST, self._finish_brief, ANALYST_MAX_TOOL_CALLS)
        self.analyst_answer = agent(ANALYST, ANALYST_TOOLS, DATA_ANALYST, self._finish_response, ANSWER_MAX_TOOL_CALLS)
        self.writer = agent(WRITER, WRITER_TOOLS, RESEARCH_WRITER, self._finish_request, WRITER_MAX_TOOL_CALLS)
        self.graph = self._build_graph()

    def run(self, ticker: str, *, force_refresh: bool = False, on_update: Callable | None = None) -> PipelineRun:
        """The research report for `ticker`, from today's cache file when one is valid."""
        self._on_update = on_update
        self._run_key = uuid4().hex[:8]  # keeps each run's agent threads apart
        start = {
            "ticker": ticker.strip().upper(), "force_refresh": force_refresh, "headlines": [], "brief": None,
            "writer_observations": [], "request": None, "response": None, "report": None, "cached": False, "warnings": [],
        }
        state = self.graph.invoke(start)
        return PipelineRun(
            run_id=self.session.run_id,
            report=ResearchReport.model_validate(state["report"]),
            brief=DataBrief.model_validate(state["brief"]),
            request=ClarificationRequest.model_validate(state["request"]),
            response=ClarificationResponse.model_validate(state["response"]),
            cached=state["cached"],
            cache_path=cache_path(self.cache_dir, state["ticker"], self.session.today),
            warnings=state["warnings"],
        )

    # --- the outer graph -------------------------------------------------------------------

    def _build_graph(self):
        graph = StateGraph(PipelineState)
        steps = [
            ("news_intake", self._news_intake),
            ("analyst", self._analyst),
            ("writer", self._writer),
            ("analyst_answer", self._analyst_answer),
            ("final_report", self._final_report),
            ("cache_save", self._cache_save),
        ]
        graph.add_node("cache_lookup", self._cache_lookup)
        for name, node in steps:
            graph.add_node(name, node)
        graph.add_edge(START, "cache_lookup")
        graph.add_conditional_edges("cache_lookup", lambda s: END if s["cached"] else "news_intake", ["news_intake", END])
        for (name, _), (following, _) in pairwise(steps):
            graph.add_edge(name, following)
        graph.add_edge("cache_save", END)
        return graph.compile()

    def _cache_lookup(self, state: PipelineState) -> dict:
        ticker = state["ticker"]
        if state["force_refresh"]:
            self.session.trace.write("cache", action="skip", reason="force_refresh")
            self._emit(PIPELINE, {"cache": {"action": "skip", "reason": "force_refresh"}})
            return {"cached": False}
        cached = load_cached(self.cache_dir, ticker, self.session.today, self.session.trace)
        if cached is None:
            self._emit(PIPELINE, {"cache": {"action": "miss", "path": str(cache_path(self.cache_dir, ticker, self.session.today))}})
            return {"cached": False}
        self._emit(PIPELINE, {"cache": {"action": "hit", "path": str(cache_path(self.cache_dir, ticker, self.session.today)), "saved_at": cached.created_at.isoformat()}})
        return {
            "cached": True,
            "brief": cached.brief.model_dump(mode="json"),
            "request": cached.request.model_dump(mode="json"),
            "response": cached.response.model_dump(mode="json"),
            "report": cached.report.model_dump(mode="json"),
        }

    def _news_intake(self, state: PipelineState) -> dict:
        result = self.session.get_news(state["ticker"], INTAKE_HEADLINES, agent=PIPELINE)
        titles = [headline.title for headline in result.data.headlines] if result.data else []
        warnings = [] if titles else [f"news intake found no headlines ({result.status}); Agent A's brief will have no sentiment"]
        self._emit(PIPELINE, {"news_intake": {"status": result.status, "headlines": titles}})
        return {"headlines": titles, "warnings": warnings}

    def _analyst(self, state: PipelineState) -> dict:
        headlines = state["headlines"]
        message = DATA_ANALYST.user.format(
            today=self.session.today, max_tool_calls=ANALYST_MAX_TOOL_CALLS, ticker=state["ticker"], count=len(headlines),
            headlines="\n".join(f"- {title}" for title in headlines) or "(none: the news intake failed)",
        )
        run = self._invoke(self.analyst, ANALYST, state["ticker"], message, {"headlines_given": len(headlines)})
        self.session.trace.write("handoff", sender=ANALYST, receiver=WRITER, schema="DataBrief", content=run.result)
        self._emit(PIPELINE, {"handoff": run.result})
        return {"brief": run.result, "warnings": run.warnings}

    def _writer(self, state: PipelineState) -> dict:
        brief = DataBrief.model_validate(state["brief"])  # the handoff is checked against its schema on receipt
        message = RESEARCH_WRITER.user.format(
            today=self.session.today, max_tool_calls=WRITER_MAX_TOOL_CALLS, ticker=state["ticker"], brief=brief.model_dump_json(),
        )
        run = self._invoke(self.writer, WRITER, state["ticker"], message, {"brief": state["brief"]})
        self.session.trace.write("critique", step="request", sender=WRITER, receiver=ANALYST, content=run.result)
        self._emit(PIPELINE, {"critique_request": run.result})
        return {"request": run.result, "writer_observations": run.observations, "warnings": run.warnings}

    def _analyst_answer(self, state: PipelineState) -> dict:
        request = ClarificationRequest.model_validate(state["request"])
        message = CLARIFICATION_TASK.format(
            today=self.session.today, max_tool_calls=ANSWER_MAX_TOOL_CALLS, ticker=state["ticker"],
            request=request.model_dump_json(), brief=DataBrief.model_validate(state["brief"]).model_dump_json(),
        )
        run = self._invoke(self.analyst_answer, ANALYST, state["ticker"], message, {"request": state["request"]}, thread="A-answer")
        self.session.trace.write("critique", step="response", sender=ANALYST, receiver=WRITER, content=run.result)
        self._emit(PIPELINE, {"clarification": run.result})
        return {"response": run.result, "warnings": run.warnings}

    def _final_report(self, state: PipelineState) -> dict:
        report = write_final_report(
            DataBrief.model_validate(state["brief"]), state["writer_observations"], WRITER_TOOLS,
            ClarificationResponse.model_validate(state["response"]), self._get_llm(), self.session.today,
        )
        self.session.trace.write(
            "report", agent=WRITER, generated_by=report.generated_by, risks=[risk.title for risk in report.top_risks],
            hedge=report.hedge.strategy, clarification_used=report.clarification_used, warnings=report.warnings,
        )
        self._emit(WRITER, {"finish": {"result": report.model_dump(mode="json")}})
        return {"report": report.model_dump(mode="json"), "warnings": report.warnings}

    def _cache_save(self, state: PipelineState) -> dict:
        if state["report"]["generated_by"] == "template":
            # A fallback report is never cached, so the next run tries the models again.
            reason = "the report was written from a template, so it is not cached"
            self.session.trace.write("cache", action="skip_save", reason=reason)
            self._emit(PIPELINE, {"cache": {"action": "skip_save", "reason": reason}})
            return {"warnings": [reason]}
        cached = CachedRun(
            schema_version=SCHEMA_VERSION, ticker=state["ticker"], as_of=self.session.today, run_id=self.session.run_id,
            profile=active_profile().name, created_at=datetime.now(timezone.utc),
            report=state["report"], brief=state["brief"], request=state["request"], response=state["response"],
        )
        path = save_cached(self.cache_dir, cached, self.session.trace)
        self._emit(PIPELINE, {"cache": {"action": "save", "path": str(path)}})
        return {}

    # --- the agents' finish steps --------------------------------------------------------------

    def _finish_brief(self, state: AgentState) -> dict:
        brief, warnings = build_brief(
            state["ticker"], self.session.today, state["observations"], state["context"]["headlines_given"], self._get_llm(),
        )
        return {"result": brief.model_dump(mode="json"), "warnings": warnings}

    def _finish_request(self, state: AgentState) -> dict:
        brief = DataBrief.model_validate(state["context"]["brief"])
        request, warnings = write_request(brief, state["observations"], self._get_llm())
        return {"result": request.model_dump(mode="json"), "warnings": warnings}

    def _finish_response(self, state: AgentState) -> dict:
        request = ClarificationRequest.model_validate(state["context"]["request"])
        answer = next((text for m in reversed(state["messages"]) if m.type == "ai" and (text := message_text(m))), "")
        response, _, warnings = build_response(request, state["ticker"], state["observations"], answer, self.session, ANALYST)
        return {"result": response.model_dump(mode="json"), "warnings": warnings}

    # --- helpers -----------------------------------------------------------------------------------

    def _invoke(self, graph, name: str, ticker: str, message: str, context: dict, *, thread: str | None = None):
        update = {"messages": [HumanMessage(message)], "observations": [], "warnings": [], "ticker": ticker, "result": None, "context": context}
        return invoke_agent(graph, name, update, f"{self._run_key}-{thread or name}", self._on_update)

    def _emit(self, agent: str, step: dict[str, Any]) -> None:
        if self._on_update:
            self._on_update(agent, step)

    def _get_llm(self) -> StructuredLLM | None:
        if self._llm is None:
            try:
                self._llm = StructuredLLM(log_dir=self.session.log_dir)
            except RuntimeError as exc:  # the primary provider's key is missing
                logger.warning("no LLM for the handoffs and report: %s", exc)
                return None
        return self._llm
