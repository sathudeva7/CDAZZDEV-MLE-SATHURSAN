"""TESTING-ONLY(openai): offline tests for the openai_dev profile in common/llm_openai.py.

Delete this file together with common/llm_openai.py. No test touches the
network: StructuredLLM gets the fake client from test_llm.py, and the agent's
chat model is only built, never called.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add a stronger paid OpenAI model (gpt-6.1-sol) for testing, in a separate file so it can be deleted before submission', Date: 2026-10-07

from __future__ import annotations

import pytest

from common.llm import StructuredLLM
from common.llm_config import PROFILES
from common.llm_openai import OPENAI_MODEL, OPENAI_PROFILE, upgraded_effort
from task3_agentic.agent import chat_models
from tests.test_llm import FALLBACK_VALUE, PROMPT, VALID, FakeClient, Sentiment


def test_the_profile_is_registered_without_a_fallback():
    profile = PROFILES[OPENAI_PROFILE]
    assert profile.primary.model == OPENAI_MODEL
    assert profile.fallback is None
    assert profile.log_file.endswith(".dev.jsonl")  # gitignored


@pytest.mark.parametrize(("asked", "sent"), [("low", "medium"), ("medium", "high"), ("high", "high")])
def test_effort_is_raised_one_step(asked, sent):
    assert upgraded_effort(asked) == sent


def test_structured_call_sends_upgraded_effort_without_temperature_or_extra_body(tmp_path):
    client = FakeClient(VALID)
    llm = StructuredLLM(log_dir=tmp_path, profile=PROFILES[OPENAI_PROFILE], client_factory=lambda provider, profile: client)

    result = llm.call(PROMPT, {"headline": "Apple beats estimates"}, Sentiment, FALLBACK_VALUE, reasoning_effort="low")

    assert result.ok and result.provider == "openai"
    request = client.requests[0]
    assert request["reasoning_effort"] == "medium"
    assert "temperature" not in request and "extra_body" not in request
    assert request["response_format"]["json_schema"]["strict"] is True


def test_agent_model_uses_the_responses_api(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    [model] = chat_models(PROFILES[OPENAI_PROFILE], effort="low")

    assert model.model_name == OPENAI_MODEL
    assert model.use_responses_api is True
    assert model.reasoning == {"effort": "medium"}
    assert model.temperature is None
