"""Offline tests for common/jev.py, with the TypeSafe SDK client replaced by a fake.

The fake answers system_one() with a scripted SDK response (built from the
JSON shape the live probe returned) or raises a scripted SDK error, and
records each request. The real API is checked by tests/test_live_analysis.py.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import json

import httpx2
import pytest
from typesafe_sdk import (
    SystemOneResponse,
    TypeSafeAuthenticationError,
    TypeSafeRateLimitError,
)

from common.jev import JEV_MODEL, ChoiceDecision, JevClient, JevQuestion
from common.llm_config import PROFILES

QUESTION = JevQuestion(
    name="headline_sentiment",
    version="1",
    instructions="Effect on the share price?",
    options={"positive": "lifts it", "negative": "weighs on it", "neutral": "no clear effect"},
    state="Headline: {headline}",
)
PROFILE = PROFILES["free"]


def answer(choice="negative", probabilities=None, confidence=0.56) -> SystemOneResponse:
    """An SDK response in the shape the live probe returned on 2026-10-06."""
    probabilities = probabilities or {"neutral": 0.29, "positive": 0.0, "negative": 0.71}
    return SystemOneResponse.model_validate(
        {
            "model": JEV_MODEL,
            "answers": {
                QUESTION.name: {"type": "choice", "choice": choice, "confidence": confidence, "probabilities": probabilities}
            },
            "usage": {"input_tokens": 384, "output_tokens": 39},
        }
    )


class FakeTypeSafe:
    """Stands in for typesafe_sdk.TypeSafeClient: returns or raises the next scripted reply."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []

    def system_one(self, state, questions, **kwargs):
        self.requests.append({"state": state, "questions": questions, **kwargs})
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def sdk_error(cls, status: int):
    return cls(status=status, body={"error": "scripted"}, headers=httpx2.Headers())


def build(tmp_path, *replies) -> tuple[JevClient, FakeTypeSafe]:
    fake = FakeTypeSafe(*replies)
    return JevClient(tmp_path, profile=PROFILE, client=fake), fake


def log_lines(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / PROFILE.log_file).read_text().splitlines()]


def test_valid_answer_is_returned_and_logged(tmp_path):
    jev, fake = build(tmp_path, answer())

    result = jev.choose(QUESTION, {"headline": "Apple CEO sells stock"})

    assert result.ok and result.error is None
    assert result.value == ChoiceDecision(choice="negative", confidence=0.56, probabilities={"neutral": 0.29, "positive": 0.0, "negative": 0.71})
    [request] = fake.requests
    assert request["state"] == "Headline: Apple CEO sells stock"
    sent = request["questions"][QUESTION.name]
    assert sent.instructions == QUESTION.instructions and dict(sent.criteria) == QUESTION.options
    [line] = log_lines(tmp_path)
    assert line["provider"] == "jev" and line["model"] == JEV_MODEL and line["outcome"] == "ok"
    assert line["prompt"] == QUESTION.name and line["prompt_version"] == "1"
    assert line["prompt_tokens"] == 384 and line["choice"] == "negative"
    assert line["probabilities"]["negative"] == 0.71


@pytest.mark.parametrize(
    ("reply", "problem"),
    [
        (answer(choice="bullish"), "not one of the options"),
        (answer(probabilities={"positive": 0.3, "negative": 0.7}), "missing probabilities for ['neutral']"),
        (answer(probabilities={"positive": 0.5, "negative": 0.5, "neutral": 0.5}), "sum to 1.50"),
    ],
)
def test_answers_that_do_not_fit_the_question_are_rejected(tmp_path, reply, problem):
    jev, _ = build(tmp_path, reply)

    result = jev.choose(QUESTION, {"headline": "h"})

    assert not result.ok and result.value is None
    assert problem in result.error
    assert log_lines(tmp_path)[0]["outcome"] == "fallback"


def test_sdk_errors_become_a_failed_result(tmp_path):
    # The SDK has already spent its own retries by the time it raises.
    jev, _ = build(tmp_path, sdk_error(TypeSafeRateLimitError, 429))

    result = jev.choose(QUESTION, {"headline": "h"})

    assert not result.ok and "TypeSafeRateLimitError" in result.error
    assert log_lines(tmp_path)[0]["errors"] == [result.error]


def test_a_bad_key_fails_every_call_without_raising(tmp_path):
    jev, fake = build(tmp_path, sdk_error(TypeSafeAuthenticationError, 401), sdk_error(TypeSafeAuthenticationError, 401))

    assert not jev.choose(QUESTION, {"headline": "a"}).ok
    assert not jev.choose(QUESTION, {"headline": "b"}).ok
    assert len(fake.requests) == 2


def test_missing_key_means_no_requests(tmp_path, monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    jev = JevClient(tmp_path, profile=PROFILE)

    result = jev.choose(QUESTION, {"headline": "h"})

    assert not result.ok and "JEV_API_KEY is not set" in result.error
    assert log_lines(tmp_path)[0]["provider"] is None
