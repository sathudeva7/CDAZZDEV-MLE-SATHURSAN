"""Structured LLM calls: one prompt in, one validated Pydantic object out.

Every LLM call in this repo goes through `StructuredLLM.call`. Each kind of
failure has exactly one owner:

- The openai SDK retries rate limits (429), server errors (5xx) and dropped
  connections on the same provider, with backoff, honouring Retry-After.
- This module retries once when an answer fails validation, sending the
  validation error back to the model (the repair retry).
- This module moves to the profile's fallback provider when a provider fails.
- When every provider has failed, the caller's fallback value comes back with
  ok=False, so a pipeline always runs to completion.

Each call appends one JSON line to <log_dir>/<profile log file>, so a reviewer
can see validation failures being caught without re-running anything.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the llm-structured-call skill with its shared helper and tests', Date: 2026-10-06
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add a stronger paid OpenAI model (gpt-6.1-sol) for testing, in a separate file so it can be deleted before submission', Date: 2026-10-07

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import openai
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from common.llm_config import Profile, Provider, active_profile, api_key
from common.llm_openai import openai_request  # TESTING-ONLY(openai)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
Outcome = Literal["ok", "repaired", "fallback"]
ReasoningEffort = Literal["low", "medium", "high"]

# Temperature 0 keeps re-runs of a notebook stable on any one model.
TEMPERATURE = 0.0
# The first answer plus one repair retry.
MAX_ATTEMPTS_PER_PROVIDER = 2
RAW_EXCERPT_CHARS = 200
# Groq's 400 error code when generated output misses the schema.
JSON_VALIDATE_FAILED = "json_validate_failed"

REPAIR_INSTRUCTION = (
    "Your previous answer did not match the required JSON schema.\n"
    "Validation error: {error}\n"
    "Reply again with only a JSON object that matches the schema."
)

# Keywords dropped from the schema sent to the provider. Groq's strict mode does
# not document support for them; Pydantic still enforces each one on the answer.
UNSENT_SCHEMA_KEYWORDS = frozenset(
    {
        "title",
        "default",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
        "format",
    }
)
# Schema keys whose children are names (field names, definition names), not keywords.
NAME_MAPPINGS = frozenset({"properties", "$defs"})


@dataclass(frozen=True)
class Prompt:
    """A prompt kept as a module-level constant.

    `name` and `version` identify it in the call log. `system` and `user` are
    separate messages; `user` is a str.format template filled at call time.
    """

    name: str
    version: str
    system: str
    user: str

    def messages(self, variables: dict[str, Any]) -> list[dict[str, str]]:
        # A missing variable raises KeyError: that is a bug in the caller, not an LLM failure.
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user.format(**variables)},
        ]


@dataclass(frozen=True)
class LLMResult(Generic[T]):
    """What every call returns. `ok` is False when `value` is the caller's fallback."""

    value: T
    ok: bool
    outcome: Outcome
    attempts: int  # requests this module sent; SDK retries are logged separately
    provider: str | None  # the provider that produced `value`; None for a fallback
    error: str | None  # the last failure seen, if any


@dataclass
class _CallStats:
    attempts: int = 0
    sdk_retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw_excerpt: str | None = None
    errors: list[str] = field(default_factory=list)


class _ProviderFailed(Exception):
    """This provider can't answer the call; move on to the next one."""


def strict_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema in the shape strict mode needs.

    Every object is closed (additionalProperties false) and lists every property
    as required. Optional fields stay required and are typed `X | None`.
    """

    def tighten(node: Any, keys_are_names: bool = False) -> Any:
        if isinstance(node, list):
            return [tighten(item) for item in node]
        if not isinstance(node, dict):
            return node
        out = {}
        for key, value in node.items():
            if not keys_are_names and key in UNSENT_SCHEMA_KEYWORDS:
                continue
            out[key] = tighten(value, keys_are_names=not keys_are_names and key in NAME_MAPPINGS)
        if not keys_are_names and out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
            out["required"] = list(out["properties"])
        return out

    return tighten(schema.model_json_schema())


ClientFactory = Callable[[Provider, Profile], Any]


def make_client(provider: Provider, profile: Profile) -> OpenAI | None:
    """An SDK client for the provider, or None when its key is not set.

    The SDK's own retries stay on: they own rate limits, 5xx and connection errors.
    """
    key = api_key(provider)
    if not key:
        return None
    return OpenAI(
        base_url=provider.base_url,
        api_key=key,
        max_retries=profile.max_retries,
        timeout=profile.timeout_s,
    )


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Start PR 4: the Task 3 notebook, laid out as settled in the grilling rounds', Date: 2026-10-07
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 data script as designed in the grilling rounds', Date: 2026-10-07
# The repo's packages whose INFO lines and warnings notebooks show.
CONSOLE_LOGGERS = ("common", "task1_financial", "task2_genai", "task3_agentic")


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 1A yfinance data pipeline and summary dictionary', Date: 2026-10-06
def enable_console_logging() -> None:
    """Show the repo's INFO lines and warnings, and the SDK's 'Retrying request' lines, in notebook output."""
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s")
    logging.getLogger("openai").setLevel(logging.INFO)
    for name in CONSOLE_LOGGERS:
        logging.getLogger(name).setLevel(logging.INFO)


class StructuredLLM:
    """Sends prompts to the active profile's providers and validates every answer.

    Create one per task, pointing `log_dir` at that task's logs/ folder.
    """

    def __init__(
        self,
        log_dir: Path | str,
        profile: Profile | None = None,
        client_factory: ClientFactory = make_client,
    ) -> None:
        self.profile = profile or active_profile()
        self.log_path = Path(log_dir) / self.profile.log_file
        self._providers: list[tuple[Provider, Any]] = []
        for provider in (self.profile.primary, self.profile.fallback):
            if provider is None:
                continue
            client = client_factory(provider, self.profile)
            if client is None:
                if provider is self.profile.primary:
                    raise RuntimeError(
                        f"{provider.api_key_env} is not set. Add it to .env locally or to Colab Secrets."
                    )
                logger.warning(
                    "%s is not set, so the %s fallback is off for this run",
                    provider.api_key_env,
                    provider.name,
                )
                continue
            self._providers.append((provider, client))

    def call(
        self,
        prompt: Prompt,
        variables: dict[str, Any],
        schema: type[T],
        fallback: T,
        *,
        reasoning_effort: ReasoningEffort = "low",
    ) -> LLMResult[T]:
        """Ask each provider in turn for an answer matching `schema`.

        Returns the first valid answer, or `fallback` with ok=False when every
        provider has failed. LLM and provider failures never raise.
        """
        messages = prompt.messages(variables)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": strict_json_schema(schema),
                "strict": True,
            },
        }
        stats = _CallStats()
        started = time.perf_counter()

        result: LLMResult[T] | None = None
        for provider, client in self._providers:
            try:
                value, repaired = self._ask(
                    provider, client, messages, schema, response_format, reasoning_effort, stats
                )
            except _ProviderFailed as exc:
                stats.errors.append(f"{provider.name}: {exc}")
                logger.warning("%s failed on %s: %s", prompt.name, provider.name, exc)
                continue
            result = LLMResult(
                value=value,
                ok=True,
                outcome="repaired" if repaired else "ok",
                attempts=stats.attempts,
                provider=provider.name,
                error=stats.errors[-1] if stats.errors else None,
            )
            break

        if result is None:
            last_error = stats.errors[-1] if stats.errors else "no provider configured"
            logger.error("%s fell back to the caller's default value: %s", prompt.name, last_error)
            result = LLMResult(
                value=fallback,
                ok=False,
                outcome="fallback",
                attempts=stats.attempts,
                provider=None,
                error=last_error,
            )

        self._log(prompt, schema, result, stats, latency_s=time.perf_counter() - started)
        return result

    def _ask(
        self,
        provider: Provider,
        client: Any,
        messages: list[dict[str, str]],
        schema: type[T],
        response_format: dict[str, Any],
        reasoning_effort: ReasoningEffort,
        stats: _CallStats,
    ) -> tuple[T, bool]:
        """One provider: the first answer plus one repair retry. Returns (value, repaired)."""
        conversation = list(messages)
        last_error = ""
        for attempt in range(1, MAX_ATTEMPTS_PER_PROVIDER + 1):
            stats.attempts += 1
            try:
                request = {
                    "model": provider.model,
                    "messages": conversation,
                    "temperature": TEMPERATURE,
                    "response_format": response_format,
                    **_reasoning_options(provider, reasoning_effort),
                }
                if provider.reasoning_style == "openai":  # TESTING-ONLY(openai)
                    request = openai_request(request, reasoning_effort)  # TESTING-ONLY(openai)
                raw_response = client.chat.completions.with_raw_response.create(**request)
            except openai.BadRequestError as exc:
                if exc.code != JSON_VALIDATE_FAILED:
                    raise _ProviderFailed(_describe_api_error(exc)) from exc
                # The provider rejected its own output against the schema; repair it like any other.
                body = exc.body if isinstance(exc.body, dict) else {}
                raw = str(body.get("failed_generation") or "")
                last_error = "output did not match the schema (json_validate_failed)"
            except openai.APIError as exc:
                # The SDK's retries are spent, or the error is permanent (bad key, unknown model).
                raise _ProviderFailed(_describe_api_error(exc)) from exc
            else:
                stats.sdk_retries += raw_response.retries_taken
                completion = raw_response.parse()
                _add_usage(stats, completion)
                if not completion.choices:
                    raise _ProviderFailed("response had no choices")
                raw = completion.choices[0].message.content or ""
                try:
                    return schema.model_validate_json(raw), attempt > 1
                except ValidationError as exc:
                    last_error = _describe_validation_error(exc)

            stats.raw_excerpt = raw[:RAW_EXCERPT_CHARS]
            logger.warning(
                "%s answer from %s failed validation (attempt %d of %d): %s",
                schema.__name__,
                provider.name,
                attempt,
                MAX_ATTEMPTS_PER_PROVIDER,
                last_error,
            )
            conversation = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": REPAIR_INSTRUCTION.format(error=last_error)},
            ]
        raise _ProviderFailed(f"invalid output after {MAX_ATTEMPTS_PER_PROVIDER} attempts: {last_error}")

    def _log(
        self,
        prompt: Prompt,
        schema: type[BaseModel],
        result: LLMResult[Any],
        stats: _CallStats,
        latency_s: float,
    ) -> None:
        provider = next((p for p, _ in self._providers if p.name == result.provider), None)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profile": self.profile.name,
            "prompt": prompt.name,
            "prompt_version": prompt.version,
            "schema": schema.__name__,
            "provider": result.provider,
            "model": provider.model if provider else None,
            "outcome": result.outcome,
            "attempts": stats.attempts,
            "sdk_retries": stats.sdk_retries,
            "latency_ms": round(latency_s * 1000),
            "prompt_tokens": stats.prompt_tokens,
            "completion_tokens": stats.completion_tokens,
            "errors": stats.errors,
            "raw_excerpt": stats.raw_excerpt,
        }
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:  # a full disk must not stop the pipeline
            logger.warning("could not write %s: %s", self.log_path, exc)


def _reasoning_options(provider: Provider, effort: ReasoningEffort) -> dict[str, Any]:
    """Request options for reasoning effort, in each provider's own shape.

    Reasoning text is left out of the reply either way; it still counts toward
    token limits, which is why effort is set per call.
    """
    if provider.reasoning_style == "groq":
        return {
            "reasoning_effort": effort,
            "extra_body": {**provider.extra_body, "include_reasoning": False},
        }
    return {"extra_body": {**provider.extra_body, "reasoning": {"effort": effort, "exclude": True}}}


def _add_usage(stats: _CallStats, completion: Any) -> None:
    usage = getattr(completion, "usage", None)
    if usage is not None:
        stats.prompt_tokens += usage.prompt_tokens or 0
        stats.completion_tokens += usage.completion_tokens or 0


def _describe_api_error(exc: openai.APIError) -> str:
    status = getattr(exc, "status_code", None)
    prefix = f"{type(exc).__name__} {status}" if status else type(exc).__name__
    return f"{prefix}: {exc.message}"


def _describe_validation_error(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)
