"""Typed decisions from Jev, TypeSafe's decision model (see docs/adr/0001).

Jev reads a text `state` and a Choice question, and answers with the chosen
option, a probability for every option, and a confidence computed from those
probabilities. It writes no text, so every explanation comes from the LLM in
common/llm.py. Each kind of failure has one owner, as in common/llm.py:

- The typesafe SDK retries rate limits (429), server errors (5xx), timeouts
  and dropped connections, with backoff, honouring Retry-After.
- The SDK validates the response body against its own Pydantic models.
- This module checks the answer against the question that was asked: the
  choice is one of the options, every option has a probability, and the
  probabilities sum to 1. Any failure comes back as ok=False, never raised,
  so the caller can fall back to the LLM.

Each request appends one JSON line to the same call log as StructuredLLM, so
one file shows every model decision of a run.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from typesafe_sdk import Choice, RetryPolicy, TypeSafeClient, TypeSafeError

from common.llm_config import Profile, active_profile, secret

logger = logging.getLogger(__name__)

JEV_API_KEY_ENV = "JEV_API_KEY"
# Pinned rather than "jev-latest", so re-running the notebook asks the same model.
JEV_MODEL = "jev-1.13.0"
JEV_PROVIDER = "jev"
JEV_MAX_RETRIES = 3
JEV_TIMEOUT_S = 30.0
# Jev rounds probabilities to two decimals, so three options can sum to 0.99 or 1.01.
PROBABILITY_SUM_TOLERANCE = 0.02


@dataclass(frozen=True)
class JevQuestion:
    """A Choice question kept as a module-level constant, like common.llm.Prompt.

    `name` and `version` identify it in the call log. `options` maps each
    answer to what it means. `state` is a str.format template for the text
    Jev judges, filled at call time.
    """

    name: str
    version: str
    instructions: str
    options: dict[str, str]
    state: str


class ChoiceDecision(BaseModel):
    """Jev's answer to one Choice question."""

    choice: str
    confidence: float = Field(ge=0, le=1, description="How far the top probability is above an even split")
    probabilities: dict[str, float] = Field(description="Probability of every option")


@dataclass(frozen=True)
class JevResult:
    """What every request returns. `value` is None, with the reason in `error`, when ok is False."""

    value: ChoiceDecision | None
    ok: bool
    error: str | None


class _BadAnswer(Exception):
    """Jev answered, but the answer does not fit the question."""


class JevClient:
    """Asks Jev Choice questions and checks every answer. Never raises for an API problem."""

    def __init__(self, log_dir: Path | str, profile: Profile | None = None, client: Any = None) -> None:
        # The call log follows the LLM profile, so paid_dev runs stay out of the submitted log.
        self.log_path = Path(log_dir) / (profile or active_profile()).log_file
        self.profile_name = (profile or active_profile()).name
        if client is None:
            key = secret(JEV_API_KEY_ENV)
            if key:
                client = TypeSafeClient(
                    api_key=key,  # passed explicitly: the SDK would otherwise read TYPESAFE_API_KEY
                    model=JEV_MODEL,
                    retry=RetryPolicy(max_retries=JEV_MAX_RETRIES, timeout=JEV_TIMEOUT_S),
                )
            else:
                logger.warning("%s is not set, so Jev is off for this run and the LLM labels every headline", JEV_API_KEY_ENV)
        self._client = client

    def choose(self, question: JevQuestion, variables: dict[str, Any]) -> JevResult:
        """Jev's checked answer to `question` about the filled-in state."""
        state = question.state.format(**variables)  # a missing variable is a caller bug, so it raises
        started = time.perf_counter()
        usage = None
        if self._client is None:
            result = JevResult(None, ok=False, error=f"{JEV_API_KEY_ENV} is not set")
        else:
            try:
                response = self._client.system_one(
                    state=state,
                    questions={question.name: Choice(instructions=question.instructions, criteria=question.options)},
                )
                usage = response.usage
                decision = _check(response.answers.get(question.name), question)
                result = JevResult(decision, ok=True, error=None)
            except TypeSafeError as exc:  # the SDK's retries are spent, or the error is permanent
                result = JevResult(None, ok=False, error=f"{type(exc).__name__}: {exc}")
            except (_BadAnswer, ValidationError) as exc:
                result = JevResult(None, ok=False, error=f"answer did not fit the question: {exc}")
        if not result.ok:
            logger.warning("Jev %s failed: %s", question.name, result.error)
        self._log(question, result, usage, latency_s=time.perf_counter() - started)
        return result

    def _log(self, question: JevQuestion, result: JevResult, usage: Any, latency_s: float) -> None:
        # Same keys as StructuredLLM's log lines where they apply, plus the decision itself.
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profile": self.profile_name,
            "prompt": question.name,
            "prompt_version": question.version,
            "schema": "Choice",
            "provider": JEV_PROVIDER if self._client is not None else None,
            "model": JEV_MODEL,
            "outcome": "ok" if result.ok else "fallback",
            "latency_ms": round(latency_s * 1000),
            "prompt_tokens": getattr(usage, "input_tokens", None),
            "completion_tokens": getattr(usage, "output_tokens", None),
            "errors": [] if result.ok else [result.error],
            "choice": result.value.choice if result.value else None,
            "probabilities": result.value.probabilities if result.value else None,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:  # a full disk must not stop the pipeline
            logger.warning("could not write %s: %s", self.log_path, exc)


def _check(answer: Any, question: JevQuestion) -> ChoiceDecision:
    """The answer as a ChoiceDecision, or _BadAnswer if it does not fit `question`."""
    if answer is None:
        raise _BadAnswer(f"no answer for {question.name!r}")
    decision = ChoiceDecision(choice=answer.choice, confidence=answer.confidence, probabilities=dict(answer.probabilities))
    options = list(question.options)
    if decision.choice not in options:
        raise _BadAnswer(f"choice {decision.choice!r} is not one of the options {options}")
    missing = [option for option in options if option not in decision.probabilities]
    if missing:
        raise _BadAnswer(f"missing probabilities for {missing}")
    total = sum(decision.probabilities[option] for option in options)
    if not math.isclose(total, 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
        raise _BadAnswer(f"probabilities sum to {total:.2f}, not 1")
    return decision
