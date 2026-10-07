"""Offline tests for Task 2's teacher provider (common/llm_openai.py).

No test touches the network: StructuredLLM gets the fake client from test_llm.py.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add a stronger paid OpenAI model (gpt-6.1-sol) for testing, in a separate file so it can be deleted before submission', Date: 2026-10-07
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'yes do both (OpenAI testing cleanup before the Task 3 submission run)', Date: 2026-10-07

from __future__ import annotations

import pytest

from common.llm import StructuredLLM
from common.llm_config import PROFILES, active_profile
from common.llm_openai import OPENAI_MODEL
from task2_genai.generate_data import TEACHER_EFFORT, TEACHER_PROFILE
from tests.test_llm import FALLBACK_VALUE, PROMPT, VALID, FakeClient, Sentiment


def test_the_teacher_is_not_a_selectable_profile(monkeypatch):
    # Tasks 1 and 3 can only run on profiles in llm_config; Sol is not one of them.
    assert all(profile.primary.model != OPENAI_MODEL for profile in PROFILES.values())
    monkeypatch.setenv("LLM_PROFILE", "openai_dev")
    with pytest.raises(ValueError):
        active_profile()


def test_teacher_profile_has_no_fallback_and_a_committed_log():
    assert TEACHER_PROFILE.primary.model == OPENAI_MODEL
    assert TEACHER_PROFILE.fallback is None
    assert TEACHER_PROFILE.log_file == "teacher_calls.jsonl"


def test_teacher_call_sends_effort_without_temperature_or_extra_body(tmp_path):
    client = FakeClient(VALID)
    llm = StructuredLLM(log_dir=tmp_path, profile=TEACHER_PROFILE, client_factory=lambda provider, profile: client)

    result = llm.call(PROMPT, {"headline": "Apple beats estimates"}, Sentiment, FALLBACK_VALUE, reasoning_effort=TEACHER_EFFORT)

    assert result.ok and result.provider == "openai"
    request = client.requests[0]
    assert request["reasoning_effort"] == "medium"
    assert "temperature" not in request and "extra_body" not in request
    assert request["response_format"]["json_schema"]["strict"] is True
