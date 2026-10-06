"""Scripted stand-ins for StructuredLLM, JevClient and the agent chat model, shared by the tests.

Each fake returns its replies in order and records every request, so a test
can check both what the code did with an answer and what it sent.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Any, ClassVar

import numpy as np
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from common.jev import ChoiceDecision, JevResult
from common.llm import LLMResult
from task3_agentic.tools import ToolSession
from task3_agentic.trace import TRACE_FILE_NAME


@dataclass
class ScriptedLLM:
    """Stands in for StructuredLLM.call. A reply is a schema instance (ok) or None (every provider failed)."""

    replies: list[Any]
    requests: list[dict] = field(default_factory=list)

    def call(self, prompt, variables, schema, fallback, *, reasoning_effort="low"):
        self.requests.append({"prompt": prompt.name, "variables": variables, "schema": schema, "effort": reasoning_effort})
        reply = self.replies.pop(0)
        if reply is None:
            return LLMResult(value=fallback, ok=False, outcome="fallback", attempts=2, provider=None, error="scripted failure")
        assert isinstance(reply, schema), f"scripted {type(reply).__name__} for a {schema.__name__} call"
        return LLMResult(value=reply, ok=True, outcome="ok", attempts=1, provider="groq", error=None)


@dataclass
class ScriptedJev:
    """Stands in for JevClient.choose. A reply is (choice, probabilities) or None (Jev failed)."""

    replies: list[Any]
    requests: list[dict] = field(default_factory=list)

    def choose(self, question, variables):
        self.requests.append({"question": question.name, "state": question.state.format(**variables)})
        reply = self.replies.pop(0)
        if reply is None:
            return JevResult(None, ok=False, error="scripted failure")
        choice, probabilities = reply
        n = len(probabilities)
        confidence = round((probabilities[choice] - 1 / n) / (1 - 1 / n), 2)  # Jev's documented formula
        return JevResult(ChoiceDecision(choice=choice, confidence=confidence, probabilities=probabilities), ok=True, error=None)


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07
class ScriptedChat(BaseChatModel):
    """Stands in for the agent's chat model. A reply is an AIMessage, or an exception to raise.

    `calls` records, per model call, the messages sent and the tool names bound
    (None when the model was called without tools).
    """

    replies: list[Any]
    calls: list[dict] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self.bind(bound_tools=[tool.name for tool in tools])

    def _generate(self, messages, stop=None, run_manager=None, bound_tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": bound_tools})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return ChatResult(generations=[ChatGeneration(message=reply)])


def tool_turn(*calls: tuple[str, dict]) -> AIMessage:
    """An AIMessage asking for each (tool name, args) call, with ids call_1, call_2, ..."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": f"call_{i}", "type": "tool_call"} for i, (name, args) in enumerate(calls, 1)],
        response_metadata={"model_name": "scripted"},
    )


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07
TODAY = date(2026, 10, 6)


def two_years_of_bars() -> pd.DataFrame:
    """A seeded random walk, so volatility changes over time and the percentile means something."""
    days = pd.bdate_range("2024-10-07", "2026-10-06", name="Date")
    rng = np.random.default_rng(7)
    close = 200.0 * np.exp(np.cumsum(rng.normal(0, 0.015, len(days))))
    return pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000.0}, index=days)


class FakeSearch:
    """Stands in for ddgs.DDGS: a reply per backend is a list of rows or an exception to raise."""

    replies: ClassVar[dict[str, object]] = {}
    queries: ClassVar[list[tuple[str, str]]] = []

    def __init__(self, timeout=None):
        pass

    def text(self, query, max_results):
        return self._reply("text", query)

    def news(self, query, max_results):
        return self._reply("news", query)

    def _reply(self, backend, query):
        FakeSearch.queries.append((backend, query))
        reply = FakeSearch.replies.get(backend, [])
        if isinstance(reply, Exception):
            raise reply
        return reply


def make_session(tmp_path, llm=None, **kwargs) -> ToolSession:
    """A Task 3 ToolSession on fixed dates, logging to tmp_path, with fake search and LLM."""
    return ToolSession(today=TODAY, subject="AAPL", log_dir=tmp_path, llm=llm or ScriptedLLM([]), search_factory=FakeSearch, **kwargs)


def trace_lines(tmp_path) -> list[dict]:
    path = tmp_path / TRACE_FILE_NAME
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
