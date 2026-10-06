"""Offline tests for common/llm.py: every failure path, with a fake client.

No test touches the network. The fake stands in for openai.OpenAI and returns
(or raises) scripted replies in order, recording each request it receives.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the llm-structured-call skill with its shared helper and tests', Date: 2026-10-06

from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Literal

import httpx2  # the HTTP client openai>=3 is built on
import openai
import pytest
from pydantic import BaseModel, Field

from common.llm import REPAIR_INSTRUCTION, Prompt, StructuredLLM, strict_json_schema
from common.llm_config import PROFILES, Profile, Provider, active_profile

REQUEST = httpx2.Request("POST", "https://provider.test/v1/chat/completions")

PRIMARY = Provider("primary", "https://primary.test/v1", "model-a", "PRIMARY_KEY", "groq")
FALLBACK = Provider(
    "fallback",
    "https://fallback.test/v1",
    "model-b",
    "FALLBACK_KEY",
    "openrouter",
    extra_body={"provider": {"require_parameters": True}},
)
WITH_FALLBACK = Profile("test", PRIMARY, FALLBACK, max_retries=0, timeout_s=1.0, log_file="calls.jsonl")
NO_FALLBACK = Profile("test", PRIMARY, None, max_retries=0, timeout_s=1.0, log_file="calls.jsonl")

PROMPT = Prompt(name="sentiment", version="1", system="Classify the headline.", user="Headline: {headline}")


class Sentiment(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1, description="How sure the model is, 0 to 1")


FALLBACK_VALUE = Sentiment(sentiment="neutral", confidence=0.0)
VALID = '{"sentiment": "positive", "confidence": 0.9}'
OUT_OF_RANGE = '{"sentiment": "positive", "confidence": 1.7}'


class FakeClient:
    """Each create() returns or raises the next scripted reply.

    A reply is an exception, a content string, or (content, sdk_retries_taken).
    """

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(with_raw_response=SimpleNamespace(create=self._create))
        )

    def _create(self, **request):
        self.requests.append(request)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content, retries = reply if isinstance(reply, tuple) else (reply, 0)
        completion = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        return SimpleNamespace(retries_taken=retries, parse=lambda: completion)


def status_error(cls, status: int, body: dict | None = None):
    return cls("provider error", response=httpx2.Response(status, request=REQUEST), body=body)


def build(tmp_path, primary: FakeClient, fallback: FakeClient | None = None, profile=WITH_FALLBACK):
    clients = {"primary": primary, "fallback": fallback}
    return StructuredLLM(tmp_path, profile=profile, client_factory=lambda p, _: clients[p.name])


def log_lines(tmp_path) -> list[dict]:
    return [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]


def test_valid_answer_is_returned_and_logged(tmp_path):
    primary = FakeClient(VALID)
    result = build(tmp_path, primary, FakeClient()).call(PROMPT, {"headline": "h"}, Sentiment, FALLBACK_VALUE)

    assert result.ok and result.outcome == "ok" and result.attempts == 1
    assert result.value == Sentiment(sentiment="positive", confidence=0.9)
    [line] = log_lines(tmp_path)
    assert line["outcome"] == "ok" and line["provider"] == "primary" and line["model"] == "model-a"
    assert line["prompt"] == "sentiment" and line["prompt_version"] == "1"
    assert line["prompt_tokens"] == 10 and line["raw_excerpt"] is None


def test_invalid_answer_is_repaired_with_the_validation_error(tmp_path):
    primary = FakeClient(OUT_OF_RANGE, VALID)
    result = build(tmp_path, primary, FakeClient()).call(PROMPT, {"headline": "h"}, Sentiment, FALLBACK_VALUE)

    assert result.ok and result.outcome == "repaired" and result.attempts == 2
    repair_turns = primary.requests[1]["messages"][-2:]
    assert repair_turns[0] == {"role": "assistant", "content": OUT_OF_RANGE}
    assert repair_turns[1]["role"] == "user" and "confidence" in repair_turns[1]["content"]
    assert repair_turns[1]["content"].startswith(REPAIR_INSTRUCTION.split("\n")[0])
    assert log_lines(tmp_path)[0]["raw_excerpt"] == OUT_OF_RANGE


def test_groq_json_validate_failed_is_repaired_from_failed_generation(tmp_path):
    rejected = status_error(
        openai.BadRequestError, 400, {"code": "json_validate_failed", "failed_generation": "{oops"}
    )
    primary = FakeClient(rejected, VALID)
    result = build(tmp_path, primary, FakeClient()).call(PROMPT, {"headline": "h"}, Sentiment, FALLBACK_VALUE)

    assert result.ok and result.outcome == "repaired"
    assert primary.requests[1]["messages"][-2] == {"role": "assistant", "content": "{oops"}


def test_two_invalid_answers_without_fallback_provider_return_the_fallback_value(tmp_path):
    primary = FakeClient(OUT_OF_RANGE, OUT_OF_RANGE)
    result = build(tmp_path, primary, profile=NO_FALLBACK).call(
        PROMPT, {"headline": "h"}, Sentiment, FALLBACK_VALUE
    )

    assert not result.ok and result.outcome == "fallback" and result.provider is None
    assert result.value == FALLBACK_VALUE and result.attempts == 2
    assert "invalid output after 2 attempts" in result.error
    line = log_lines(tmp_path)[0]
    assert line["outcome"] == "fallback" and line["model"] is None and len(line["errors"]) == 1


def test_rate_limit_after_sdk_retries_moves_to_fallback_provider(tmp_path):
    primary = FakeClient(status_error(openai.RateLimitError, 429))
    fallback = FakeClient(VALID)
    result = build(tmp_path, primary, fallback).call(PROMPT, {"headline": "h"}, Sentiment, FALLBACK_VALUE)

    assert result.ok and result.provider == "fallback" and result.attempts == 2
    assert "RateLimitError 429" in result.error
    assert log_lines(tmp_path)[0]["errors"][0].startswith("primary: RateLimitError 429")


@pytest.mark.parametrize(
    "error",
    [
        status_error(openai.AuthenticationError, 401),
        status_error(openai.BadRequestError, 400, {"code": "model_not_found"}),
        openai.APIConnectionError(request=REQUEST),
    ],
)
def test_permanent_and_connection_errors_skip_the_repair_retry(tmp_path, error):
    primary = FakeClient(error)
    fallback = FakeClient(VALID)
    result = build(tmp_path, primary, fallback).call(PROMPT, {"headline": "h"}, Sentiment, FALLBACK_VALUE)

    assert result.ok and result.provider == "fallback"
    assert len(primary.requests) == 1


def test_every_provider_failing_returns_the_fallback_value(tmp_path, caplog):
    primary = FakeClient(status_error(openai.InternalServerError, 503))
    fallback = FakeClient(OUT_OF_RANGE, OUT_OF_RANGE)
    with caplog.at_level(logging.WARNING, logger="common.llm"):
        result = build(tmp_path, primary, fallback).call(
            PROMPT, {"headline": "h"}, Sentiment, FALLBACK_VALUE
        )

    assert not result.ok and result.value == FALLBACK_VALUE and result.attempts == 3
    assert len(log_lines(tmp_path)[0]["errors"]) == 2
    assert "fell back to the caller's default value" in caplog.text


def test_sdk_retries_are_counted_in_the_log(tmp_path):
    primary = FakeClient((VALID, 2))
    build(tmp_path, primary, FakeClient()).call(PROMPT, {"headline": "h"}, Sentiment, FALLBACK_VALUE)

    assert log_lines(tmp_path)[0]["sdk_retries"] == 2


def test_request_shape_for_each_provider(tmp_path):
    primary = FakeClient(status_error(openai.RateLimitError, 429))
    fallback = FakeClient(VALID)
    build(tmp_path, primary, fallback).call(
        PROMPT, {"headline": "Rates cut"}, Sentiment, FALLBACK_VALUE, reasoning_effort="medium"
    )

    groq_request, openrouter_request = primary.requests[0], fallback.requests[0]
    assert groq_request["temperature"] == 0
    assert groq_request["messages"] == [
        {"role": "system", "content": "Classify the headline."},
        {"role": "user", "content": "Headline: Rates cut"},
    ]
    assert groq_request["response_format"]["json_schema"]["strict"] is True
    assert groq_request["reasoning_effort"] == "medium"
    assert groq_request["extra_body"] == {"include_reasoning": False}
    assert "reasoning_effort" not in openrouter_request
    assert openrouter_request["extra_body"] == {
        "provider": {"require_parameters": True},
        "reasoning": {"effort": "medium", "exclude": True},
    }


def test_strict_schema_closes_every_object_and_keeps_field_names():
    class Source(BaseModel):
        title: str  # a field named like a dropped keyword must survive
        url: str | None = None

    class Brief(BaseModel):
        score: float = Field(ge=-1, le=1, description="Overall score")
        sources: list[Source]

    schema = strict_json_schema(Brief)
    source = schema["$defs"]["Source"]

    assert schema["additionalProperties"] is False and schema["required"] == ["score", "sources"]
    assert source["additionalProperties"] is False and source["required"] == ["title", "url"]
    assert "title" in source["properties"]
    assert schema["properties"]["score"] == {"description": "Overall score", "type": "number"}
    assert "default" not in json.dumps(schema) and '"title": "Brief"' not in json.dumps(schema)


def test_missing_primary_key_fails_loudly(tmp_path):
    with pytest.raises(RuntimeError, match="PRIMARY_KEY is not set"):
        StructuredLLM(tmp_path, profile=WITH_FALLBACK, client_factory=lambda p, _: None)


def test_missing_fallback_key_turns_the_fallback_off(tmp_path, caplog):
    primary = FakeClient(status_error(openai.RateLimitError, 429))
    factory = lambda p, _: primary if p.name == "primary" else None
    with caplog.at_level(logging.WARNING, logger="common.llm"):
        llm = StructuredLLM(tmp_path, profile=WITH_FALLBACK, client_factory=factory)
        result = llm.call(PROMPT, {"headline": "h"}, Sentiment, FALLBACK_VALUE)

    assert "FALLBACK_KEY is not set" in caplog.text
    assert not result.ok


def test_missing_prompt_variable_is_a_bug_not_a_fallback(tmp_path):
    with pytest.raises(KeyError):
        build(tmp_path, FakeClient(VALID), FakeClient()).call(PROMPT, {}, Sentiment, FALLBACK_VALUE)


def test_profile_selection(monkeypatch):
    monkeypatch.delenv("LLM_PROFILE", raising=False)
    assert active_profile().name == "free"
    monkeypatch.setenv("LLM_PROFILE", "paid_dev")
    assert active_profile().fallback is None
    monkeypatch.setenv("LLM_PROFILE", "nope")
    with pytest.raises(ValueError, match="free"):
        active_profile()
    assert PROFILES["paid_dev"].log_file.endswith(".dev.jsonl")
