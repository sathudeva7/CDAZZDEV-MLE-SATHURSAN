"""Scripted stand-ins for StructuredLLM and JevClient, shared by the Task 1B tests.

Each fake returns its replies in order and records every request, so a test
can check both what the code did with an answer and what it sent.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from common.jev import ChoiceDecision, JevResult
from common.llm import LLMResult


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
