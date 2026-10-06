"""Provider profiles: which endpoint and model every structured LLM call uses.

The LLM_PROFILE environment variable picks a profile. The default is `free`,
the profile submitted notebooks run on. `paid_dev` is for local testing only:
it calls the same model on Groq's paid tier, so moving back to `free` changes
the rate limits and nothing else.

Keys come from the environment (a gitignored .env locally) or from Colab
Secrets in a notebook. They never appear in code.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the llm-structured-call skill with its shared helper and tests', Date: 2026-10-06

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal

from dotenv import load_dotenv

load_dotenv()  # finds the repo-root .env when running locally; a no-op in Colab

# How a provider takes the reasoning-effort setting. Groq accepts the OpenAI
# `reasoning_effort` parameter for gpt-oss; OpenRouter wants `reasoning: {effort}`.
ReasoningStyle = Literal["groq", "openrouter"]

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# gpt-oss-120b is Groq's free-tier replacement for Llama-3.3-70B (deprecated 2026-08-16).
GROQ_MODEL = "openai/gpt-oss-120b"
# The strongest OpenRouter free model with structured-output support (checked 2026-10-06).
# Free OpenRouter use is capped at 50 requests a day, so it is a fallback only.
OPENROUTER_FALLBACK_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# The SDK's default timeout is 600 s, which would freeze a notebook cell for ten minutes.
REQUEST_TIMEOUT_S = 60.0
# The free tier's 429 Retry-After is usually under a minute; four SDK retries ride it out.
FREE_MAX_RETRIES = 4
PAID_MAX_RETRIES = 2

DEFAULT_PROFILE = "free"


@dataclass(frozen=True)
class Provider:
    """One OpenAI-compatible endpoint and the model called on it."""

    name: str
    base_url: str
    model: str
    api_key_env: str
    reasoning_style: ReasoningStyle
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Profile:
    """A primary provider, an optional fallback, and the SDK retry settings."""

    name: str
    primary: Provider
    fallback: Provider | None
    max_retries: int
    timeout_s: float
    log_file: str


GROQ_FREE = Provider("groq", GROQ_BASE_URL, GROQ_MODEL, "GROQ_API_KEY", "groq")
# A separate key, so the free key stays on a free-tier account and `free` runs
# meet the real free limits.
GROQ_PAID = Provider("groq-paid", GROQ_BASE_URL, GROQ_MODEL, "GROQ_DEV_API_KEY", "groq")
OPENROUTER_FREE = Provider(
    "openrouter",
    OPENROUTER_BASE_URL,
    OPENROUTER_FALLBACK_MODEL,
    "OPENROUTER_API_KEY",
    "openrouter",
    # Route only to endpoints that enforce response_format, rather than ones that ignore it.
    extra_body={"provider": {"require_parameters": True}},
)

PROFILES: dict[str, Profile] = {
    "free": Profile(
        name="free",
        primary=GROQ_FREE,
        fallback=OPENROUTER_FREE,
        max_retries=FREE_MAX_RETRIES,
        timeout_s=REQUEST_TIMEOUT_S,
        log_file="llm_calls.jsonl",
    ),
    # No fallback: while testing, a failure should surface rather than be absorbed.
    "paid_dev": Profile(
        name="paid_dev",
        primary=GROQ_PAID,
        fallback=None,
        max_retries=PAID_MAX_RETRIES,
        timeout_s=REQUEST_TIMEOUT_S,
        log_file="llm_calls.dev.jsonl",  # gitignored: only free-profile logs are submitted
    ),
}


def active_profile() -> Profile:
    """The profile named by LLM_PROFILE, defaulting to `free`."""
    name = os.environ.get("LLM_PROFILE", DEFAULT_PROFILE)
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"LLM_PROFILE={name!r} is not one of {sorted(PROFILES)}") from None


def api_key(provider: Provider) -> str | None:
    """The provider's key from the environment, then from Colab Secrets; None if unset."""
    value = os.environ.get(provider.api_key_env)
    if value:
        return value
    try:
        from google.colab import userdata  # only importable inside Colab
    except ImportError:
        return None
    try:
        return userdata.get(provider.api_key_env)
    except (userdata.SecretNotFoundError, userdata.NotebookAccessError):
        return None
