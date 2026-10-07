"""OpenAI's GPT-6.1 Sol as a provider, used only by Task 2's teacher (task2_genai/generate_data.py).

Sol is a paid API, a deliberate exception to the free-tier setup
(docs/adr/0002-paid-teacher-for-task2.md). It is not a profile in
common/llm_config.py, so LLM_PROFILE cannot select it and no notebook or agent
runs on it: Tasks 1 and 3 run on the `free` profile.

OpenAI's reasoning models take a different request shape from Groq and
OpenRouter: effort goes in the top-level `reasoning_effort`, with no
`extra_body`, and no `temperature`, which they do not document.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add a stronger paid OpenAI model (gpt-6.1-sol) for testing, in a separate file so it can be deleted before submission', Date: 2026-10-07
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'yes do both (OpenAI testing cleanup before the Task 3 submission run)', Date: 2026-10-07

from __future__ import annotations

from typing import Any

from common.llm_config import Provider

OPENAI_BASE_URL = "https://api.openai.com/v1"
# Near-Astra quality at a fifth of the price (checked 2026-10-07 on developers.openai.com/api/docs/models).
OPENAI_MODEL = "gpt-6.1-sol"
# Medium effort can think for longer than the shared 60 s timeout.
OPENAI_TIMEOUT_S = 180.0

OPENAI = Provider("openai", OPENAI_BASE_URL, OPENAI_MODEL, "OPENAI_API_KEY", "openai")


def openai_request(request: dict[str, Any], effort: str) -> dict[str, Any]:
    """A Chat Completions request rewritten for OpenAI: top-level effort, no temperature or extra_body."""
    request = {key: value for key, value in request.items() if key not in ("temperature", "extra_body")}
    request["reasoning_effort"] = effort
    return request
