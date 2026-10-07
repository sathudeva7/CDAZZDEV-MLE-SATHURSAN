"""The tool-calling agent loop, written as a small LangGraph StateGraph.

    START -> agent --(tool calls)--> tools --(budget left)--> agent
               |                       |
               |                       +--(budget used up)--> finish -> END
               +--(no tool calls)--> finish (first time) or END (follow-up)

- agent: one chat-model turn with this agent's tools bound. The tool list is
  what the model is *given*, so restriction is enforced by binding, not by
  the prompt. Once the tool budget is used up, the model is called without
  tools and has to answer in text.
- tools: runs each requested call through ToolSession.call. It is not
  LangGraph's prebuilt ToolNode, so that one node can enforce the budget,
  refuse a tool this agent doesn't have, turn bad arguments into an error the
  model can read, and store each call as a JSON observation. Calls run one at
  a time, so the trace order is the decision order.
- finish: the caller's closing step (the 3A report, or the 3B handoff). It
  runs once per thread; a follow-up question on the same thread ends at END
  with the model's text answer.

Budgets per invocation: MAX_TOOL_CALLS tool calls and MAX_TURNS model turns,
both reset by each new question. State holds only JSON-friendly values
(messages, observation dicts, the result as a dict), because the in-memory
checkpointer that gives short-term memory turns other objects into dicts.

Model: ChatOpenAI built from active_profile(), the same endpoints, keys and
retries as StructuredLLM, with the profile's fallback provider attached via
with_fallbacks. The SDK retries 429s, honouring Retry-After, before the
fallback is tried.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add a stronger paid OpenAI model (gpt-6.1-sol) for testing, in a separate file so it can be deleted before submission', Date: 2026-10-07

from __future__ import annotations

import json
import logging
import operator
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import ValidationError

from common.llm import StructuredLLM
from common.llm_config import Profile, Provider, active_profile, api_key
from common.llm_openai import openai_chat_model  # TESTING-ONLY(openai)
from task3_agentic.prompts import RESEARCH_AGENT, RESEARCH_QUERY
from task3_agentic.report import write_report
from task3_agentic.schemas import ResearchReport, ToolResult
from task3_agentic.tools import TOOL_NAMES, WHY_ARG, ToolSession, split_agent_args

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 8
MAX_TURNS = 6
AGENT_EFFORT = "low"  # choosing the next tool is a small decision; reasoning tokens count against 8K a minute
RECURSION_LIMIT = 4 * MAX_TURNS  # LangGraph's step cap: well above the 2 x MAX_TURNS + 1 steps a run can take
SINGLE_AGENT = "single"
TRACE_TEXT_CHARS = 200
NO_BUDGET_NOTE = "Your tool budget for this question is used up. Answer from what you have, in plain text."

AGENT_NODE, TOOLS_NODE, FINISH_NODE = "agent", "tools", "finish"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    observations: Annotated[list[dict], operator.add]  # one per tool call, kept across the thread
    warnings: Annotated[list[str], operator.add]
    ticker: str
    tool_calls_used: int  # reset by each new question
    turns: int  # reset by each new question
    result: dict | None  # what finish produced; None until it runs
    context: dict  # JSON inputs for the finish step from the caller (3B: the request being answered)


@dataclass
class AgentRun:
    """What one research run or follow-up question produced."""

    thread_id: str
    answer: str  # the model's last text message
    observations: list[dict]  # this invocation's tool calls only
    report: ResearchReport | None  # 3A's report; None for the 3B agents, whose output is `result`
    warnings: list[str] = field(default_factory=list)
    result: dict | None = None  # what the finish step produced, as JSON

    @property
    def tool_calls(self) -> int:
        """Tool calls that actually ran in this invocation."""
        return sum(obs["result"] is not None for obs in self.observations)


# --- the chat model -------------------------------------------------------------


def chat_models(profile: Profile | None = None, effort: str = AGENT_EFFORT) -> list[BaseChatModel]:
    """The profile's chat models, primary first. Raises RuntimeError when the primary key is missing."""
    profile = profile or active_profile()
    models: list[BaseChatModel] = []
    for provider in (profile.primary, profile.fallback):
        if provider is None:
            continue
        key = api_key(provider)
        if key is None:
            if provider is profile.primary:
                raise RuntimeError(f"{provider.api_key_env} is not set. Add it to .env locally or to Colab Secrets.")
            logger.warning("%s is not set, so the agent's %s fallback is off for this run", provider.api_key_env, provider.name)
            continue
        models.append(_chat_model(provider, profile, key, effort))
    return models


def _chat_model(provider: Provider, profile: Profile, key: str, effort: str) -> ChatOpenAI:
    if provider.reasoning_style == "openai":  # TESTING-ONLY(openai)
        return openai_chat_model(provider, profile, key, effort)  # TESTING-ONLY(openai)
    # Groq takes reasoning_effort directly; OpenRouter wants it under `reasoning` (see common/llm.py).
    options: dict[str, Any] = {"reasoning_effort": effort} if provider.reasoning_style == "groq" else {}
    extra_body = dict(provider.extra_body)
    if provider.reasoning_style == "openrouter":
        extra_body["reasoning"] = {"effort": effort}
    return ChatOpenAI(
        base_url=provider.base_url,
        model=provider.model,
        api_key=key,
        max_retries=profile.max_retries,
        timeout=profile.timeout_s,
        temperature=0,
        extra_body=extra_body or None,
        **options,
    )


def with_fallbacks(models: Sequence[BaseChatModel], tools: Sequence[Any] = ()) -> Runnable:
    """The first model with `tools` bound, falling back to each later model, also with `tools` bound."""
    bound = [model.bind_tools(tools) if tools else model for model in models]
    return bound[0].with_fallbacks(bound[1:]) if len(bound) > 1 else bound[0]


# --- the graph ----------------------------------------------------------------------


def build_agent_graph(
    *,
    session: ToolSession,
    tool_names: Sequence[str],
    agent_name: str,
    models: Sequence[BaseChatModel],
    system_prompt: str,
    finish: Callable[[AgentState], dict],
    checkpointer: Any = None,
    max_tool_calls: int = MAX_TOOL_CALLS,
    max_turns: int = MAX_TURNS,
):
    """The compiled agent loop for one agent: its tools, its prompt and its closing step."""
    tool_names = tuple(tool_names)
    model_with_tools = with_fallbacks(models, session.langchain_tools(tool_names, agent=agent_name))
    model_alone = with_fallbacks(models)

    def agent(state: AgentState) -> dict:
        turn = state["turns"] + 1
        can_call = state["tool_calls_used"] < max_tool_calls and turn <= max_turns
        messages = [SystemMessage(system_prompt), *state["messages"]]
        if not can_call:
            messages.append(HumanMessage(NO_BUDGET_NOTE))  # sent once, not stored in the thread
        warnings = []
        try:
            reply = (model_with_tools if can_call else model_alone).invoke(messages)
        # Every provider failing must end the run with a fallback report, not a traceback.
        except Exception as exc:  # noqa: BLE001
            message = f"agent model unavailable on turn {turn}: {type(exc).__name__}: {exc}"
            logger.warning(message)
            warnings.append(message)
            reply = AIMessage(content=f"[{message}]")
        session.trace.write(
            "agent_turn",
            agent=agent_name,
            turn=turn,
            model=reply.response_metadata.get("model_name"),
            tool_calls=[call["name"] for call in reply.tool_calls],
            text=message_text(reply)[:TRACE_TEXT_CHARS],
            tokens=(reply.usage_metadata or {}).get("total_tokens"),
        )
        return {"messages": [reply], "turns": turn, "warnings": warnings}

    def tools(state: AgentState) -> dict:
        used = state["tool_calls_used"]
        messages, observations = [], []
        for call in state["messages"][-1].tool_calls:
            content, observation, ran = _run_call(session, call, tool_names, agent_name, budget_left=used < max_tool_calls)
            used += ran
            messages.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
            observations.append(observation)
        return {"messages": messages, "observations": observations, "tool_calls_used": used}

    def after_agent(state: AgentState) -> str:
        if state["messages"][-1].tool_calls:
            return TOOLS_NODE
        return FINISH_NODE if state["result"] is None else END

    def after_tools(state: AgentState) -> str:
        budget_left = state["tool_calls_used"] < max_tool_calls and state["turns"] < max_turns
        if budget_left or state["result"] is not None:
            return AGENT_NODE  # a follow-up gets one tool-less turn to answer
        return FINISH_NODE

    graph = StateGraph(AgentState)
    graph.add_node(AGENT_NODE, agent)
    graph.add_node(TOOLS_NODE, tools)
    graph.add_node(FINISH_NODE, finish)
    graph.add_edge(START, AGENT_NODE)
    graph.add_conditional_edges(AGENT_NODE, after_agent, [TOOLS_NODE, FINISH_NODE, END])
    graph.add_conditional_edges(TOOLS_NODE, after_tools, [AGENT_NODE, FINISH_NODE])
    graph.add_edge(FINISH_NODE, END)
    return graph.compile(checkpointer=checkpointer)


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 3: the 3B two-agent pipeline with the critique loop and the persistent cache, as designed in the grilling rounds', Date: 2026-10-07
def make_observation(tool: str, args: dict, why: str | None, result: ToolResult) -> dict:
    """The JSON record of one tool call that ran: what the agent asked, why, and what came back."""
    return {"tool": tool, "args": args, "why": why, "status": result.status, "digest": result.for_llm(), "result": result.model_dump(mode="json")}


def _run_call(session: ToolSession, call: dict, tool_names: Sequence[str], agent: str, *, budget_left: bool) -> tuple[str, dict, bool]:
    """One requested tool call: (text for the model, observation, whether a tool actually ran)."""
    name, raw = call["name"], dict(call.get("args") or {})
    why = raw.get(WHY_ARG)
    if name not in tool_names:
        status, problem = "refused", f"{name} is not one of your tools; you have: {', '.join(tool_names)}"
    elif not budget_left:
        status, problem = "skipped", "not run: the tool budget for this question is used up"
    else:
        try:
            args, why = split_agent_args(name, raw)
        except ValidationError as exc:
            status, problem = "invalid_args", f"bad arguments: {_describe(exc)}"
        else:
            result = session.call(name, args, agent=agent, available=tool_names, why=why)
            observation = make_observation(name, args, why, result)
            return observation["digest"], observation, True

    # The call never reached a tool: tell the model why, and trace it like any other call.
    digest = json.dumps({"tool": name, "status": "error", "error": problem})
    logger.warning("%s call %s: %s", agent, status, problem)
    args = {key: value for key, value in raw.items() if key != WHY_ARG}
    session.trace.tool_call(agent=agent, tool=name, args=args, status=status, output=digest, duration_ms=0.0, cache_hit=False, why=why)
    return digest, {"tool": name, "args": args, "why": why, "status": status, "digest": digest, "result": None}, False


# --- Task 3A: the single research agent ----------------------------------------------------


class ResearchAgent:
    """Task 3A: one agent with all five tools, a report at the end, and memory per thread.

    `research(ticker)` runs the brief's query and writes the report.
    `ask(question, thread_id)` continues the same conversation: the earlier
    tool results are still in its messages, so it can answer without new calls.
    """

    def __init__(
        self,
        session: ToolSession,
        *,
        models: Sequence[BaseChatModel] | None = None,
        llm: StructuredLLM | None = None,
        tool_names: Sequence[str] = TOOL_NAMES,
        name: str = SINGLE_AGENT,
        checkpointer: Any = None,
    ) -> None:
        self.session = session
        self.name = name
        self.tool_names = tuple(tool_names)
        self._llm = llm
        self.graph = build_agent_graph(
            session=session,
            tool_names=self.tool_names,
            agent_name=name,
            models=models or chat_models(),
            system_prompt=RESEARCH_AGENT.system,
            finish=self._finish,
            checkpointer=checkpointer or InMemorySaver(),
        )

    def research(self, ticker: str, *, thread_id: str | None = None, on_update: Callable | None = None) -> AgentRun:
        ticker = ticker.strip().upper()
        task = RESEARCH_QUERY.format(ticker=ticker)
        message = RESEARCH_AGENT.user.format(today=self.session.today, max_tool_calls=MAX_TOOL_CALLS, task=task)
        start = {"messages": [HumanMessage(message)], "observations": [], "warnings": [], "ticker": ticker, "result": None}
        return self._run(start, thread_id or uuid4().hex[:12], on_update)

    def ask(self, question: str, *, thread_id: str, on_update: Callable | None = None) -> AgentRun:
        """A follow-up in an existing thread; the report is not rewritten."""
        return self._run({"messages": [HumanMessage(question)]}, thread_id, on_update)

    def _run(self, update: dict, thread_id: str, on_update: Callable | None) -> AgentRun:
        run = invoke_agent(self.graph, self.name, update, thread_id, on_update)
        run.report = ResearchReport.model_validate(run.result) if run.result else None
        return run

    def _finish(self, state: AgentState) -> dict:
        report = write_report(state["ticker"], self.session.today, state["observations"], self._get_llm(), self.tool_names)
        self.session.trace.write(
            "report",
            agent=self.name,
            generated_by=report.generated_by,
            risks=[risk.title for risk in report.top_risks],
            hedge=report.hedge.strategy,
            warnings=report.warnings,
        )
        return {"result": report.model_dump(mode="json"), "warnings": report.warnings}

    def _get_llm(self) -> StructuredLLM | None:
        if self._llm is None:
            try:
                self._llm = StructuredLLM(log_dir=self.session.log_dir)
            except RuntimeError as exc:  # the primary provider's key is missing
                logger.warning("no LLM for the report: %s", exc)
                return None
        return self._llm


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 3: the 3B two-agent pipeline with the critique loop and the persistent cache, as designed in the grilling rounds', Date: 2026-10-07
def invoke_agent(graph, name: str, update: dict, thread_id: str, on_update: Callable | None = None) -> AgentRun:
    """Run one question through a compiled agent graph, streaming each step to `on_update(name, step)`.

    Shared by the 3A ResearchAgent and the 3B pipeline's agents. Each call gets
    a fresh tool and turn budget; observations and warnings are this call's only.
    """
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}
    previous = graph.get_state(config).values
    seen_obs, seen_warnings = len(previous.get("observations", [])), len(previous.get("warnings", []))
    update = {**update, "tool_calls_used": 0, "turns": 0}

    for step in graph.stream(update, config, stream_mode="updates"):
        if on_update:
            on_update(name, step)

    state = graph.get_state(config).values
    return AgentRun(
        thread_id=thread_id,
        answer=message_text(next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), AIMessage(""))),
        observations=state["observations"][seen_obs:],
        report=None,
        warnings=state["warnings"][seen_warnings:],
        result=state.get("result"),
    )


def message_text(message: AIMessage) -> str:
    """The message's text, whether the provider sent a string or a list of content blocks."""
    if isinstance(message.content, str):
        return message.content
    return "".join(block.get("text", "") for block in message.content if isinstance(block, dict))


def _describe(exc: ValidationError) -> str:
    return "; ".join(f"{'.'.join(str(p) for p in error['loc'])}: {error['msg']}" for error in exc.errors())
