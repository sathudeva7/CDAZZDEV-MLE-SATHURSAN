"""TESTING-ONLY(openai): an `openai_dev` profile that runs everything on OpenAI's gpt-6.1-sol.

Added so the candidate can judge Task 3's output on a stronger paid model
before choosing the submission profile. It is meant to be deleted before
submission (CLAUDE.md rule 5: submitted runs are free tier). To remove it,
delete this file and every line `grep -rn "TESTING-ONLY(openai)"` finds.

Two things differ from the Groq and OpenRouter providers:

- Tool calling: OpenAI's reasoning guide says Chat Completions does not support
  function calling with gpt-6.1-sol, so the agents' chat model uses the
  Responses API (`use_responses_api=True`). StructuredLLM sends no tools, so
  it stays on Chat Completions with a strict json_schema response_format.
- Request shape: effort goes in the top-level `reasoning_effort` (Chat
  Completions) or `reasoning.effort` (Responses), with no `extra_body`, and no
  `temperature`, which reasoning models do not document.

Effort is raised one step on this profile only (low -> medium, medium -> high):
the call sites keep the free tier's settings, which stay tuned for Groq's
8K-tokens-a-minute limit.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add a stronger paid OpenAI model (gpt-6.1-sol) for testing, in a separate file so it can be deleted before submission', Date: 2026-10-07

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from common.llm_config import PAID_MAX_RETRIES, PROFILES, Profile, Provider

if TYPE_CHECKING:
    from langchain_openai import ChatOpenAI

OPENAI_PROFILE = "openai_dev"
OPENAI_BASE_URL = "https://api.openai.com/v1"
# Near-Astra quality at a fifth of the price (checked 2026-10-07 on developers.openai.com/api/docs/models).
OPENAI_MODEL = "gpt-6.1-sol"
# High effort can think for well over the shared 60 s timeout.
OPENAI_TIMEOUT_S = 180.0
EFFORT_UPGRADE = {"low": "medium", "medium": "high", "high": "high"}

OPENAI = Provider("openai", OPENAI_BASE_URL, OPENAI_MODEL, "OPENAI_API_KEY", "openai")

PROFILES[OPENAI_PROFILE] = Profile(
    name=OPENAI_PROFILE,
    primary=OPENAI,
    fallback=None,  # like paid_dev: while testing, a failure should surface, not switch models
    max_retries=PAID_MAX_RETRIES,
    timeout_s=OPENAI_TIMEOUT_S,
    log_file="llm_calls.openai.dev.jsonl",  # gitignored by *.dev.jsonl
)


def upgraded_effort(effort: str) -> str:
    """The effort this profile actually sends: one step above the free tier's."""
    return EFFORT_UPGRADE.get(effort, effort)


def openai_request(request: dict[str, Any], effort: str) -> dict[str, Any]:
    """A Chat Completions request rewritten for OpenAI: upgraded effort, no temperature or extra_body."""
    request = {key: value for key, value in request.items() if key not in ("temperature", "extra_body")}
    request["reasoning_effort"] = upgraded_effort(effort)
    return request


def openai_chat_model(provider: Provider, profile: Profile, key: str, effort: str) -> ChatOpenAI:
    """The agents' tool-calling model, on the Responses API."""
    from langchain_openai import (
        ChatOpenAI,  # imported here: common/ is also used by Task 1, which has no LangChain
    )

    return ChatOpenAI(
        base_url=provider.base_url,
        model=provider.model,
        api_key=key,
        max_retries=profile.max_retries,
        timeout=profile.timeout_s,
        use_responses_api=True,
        reasoning={"effort": upgraded_effort(effort)},
    )
