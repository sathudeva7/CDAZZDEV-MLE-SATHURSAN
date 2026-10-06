---
name: llm-structured-call
description: Add or change an LLM call in this repo through common/llm.py, using a Prompt constant, a Pydantic schema and StructuredLLM.call with a fallback. Use when writing code that calls an LLM (headline sentiment, signal reasoning, the llm_sentiment tool, a judge), wiring Task 3's agent model, or switching LLM profile, provider or model.
---

# Structured LLM calls

Every LLM call goes through `StructuredLLM.call` in `common/llm.py`. Its module
docstring says which failure each layer owns: SDK retries, the repair retry,
provider fallback, and the caller's fallback value. Read it before changing
how failures are handled. Task 1B's rubric scores exactly what this path
produces: validated JSON, logged failures, and separate system and user roles.

## Adding a call

1. **Schema**, in the task's `schemas.py`:
   - `Literal[...]` for fixed choices, and `Field(ge=..., le=...)` for bounded numbers.
   - A `description` on every field. It goes to the model inside the JSON
     schema and guides its answer.
   - Every field is required. An optional value is `X | None` with no default,
     because strict mode needs every property in `required`.
2. **Prompt**, in the task's `prompts.py`, as a module-level constant:
   `HEADLINE_SENTIMENT = Prompt(name="headline_sentiment", version="1", system=..., user="Headline: {headline}")`.
   - The system message carries the role, the rules and what each field means.
   - The user message carries only the data, as `{placeholders}`.
   - Bump `version` whenever the text changes, so the call log ties each
     answer to the prompt that produced it.
3. **Call.** Each task has one client:
   `LLM = StructuredLLM(log_dir=Path(__file__).parent / "logs")`.
   Then `result = LLM.call(PROMPT, {...}, Schema, fallback=..., reasoning_effort=...)`.
   - `fallback` is a valid instance that downstream code can recognise, for
     example `neutral` with confidence `0.0`.
   - Branch on `result.ok`: leave fallbacks out of aggregate scores, and label
     them in reports.
4. **Reasoning effort.** Use `low` for classification and extraction, and
   `medium` when the call reasons across several signals. `high` needs a
   comment explaining why, because reasoning tokens count against the free
   tier's 8K tokens per minute.
5. **Test** the caller's handling of `ok=False` offline, with the `FakeClient`
   pattern from `tests/test_llm.py`.

Done when `pytest -q` passes, the prompt and schema live in the task's own
files, and every `LLM.call` result is checked for `ok`.

## Profiles and keys

`LLM_PROFILE` picks a profile from `common/llm_config.py`:

- **`free`** is the default and the only profile a submitted run uses. It
  calls Groq with `GROQ_API_KEY` and falls back to OpenRouter with
  `OPENROUTER_API_KEY`. It logs to `logs/llm_calls.jsonl`, which is committed.
- **`paid_dev`** is for local testing. It calls the same model with
  `GROQ_DEV_API_KEY`, has no fallback, and logs to `llm_calls.dev.jsonl`,
  which is gitignored.

Keys come from `.env` locally or from Colab Secrets, read by `api_key()`.
Notebooks call `enable_console_logging()` once, so the SDK's
"Retrying request" lines, the validation warnings and the data pipeline's
fetch and fallback lines show in the cell output.

To change a model or provider, edit `common/llm_config.py` only. Free model
lists change often, so check console.groq.com/docs/models and
openrouter.ai/api/v1/models first, and update the "checked" date in the
comment. Free OpenRouter models need the data-policy setting enabled at
openrouter.ai/settings/privacy.

## Task 3's agent model

Groq's strict structured output can't be combined with tool calling, so the
agents' tool-calling chat model is a LangChain `ChatOpenAI` built from the
same profile:

```python
profile = active_profile()
chat = ChatOpenAI(
    base_url=profile.primary.base_url,
    model=profile.primary.model,
    api_key=api_key(profile.primary),
    max_retries=profile.max_retries,
    timeout=profile.timeout_s,
    temperature=0,
)
```

The `llm_sentiment` tool and every other structured answer still go through
`StructuredLLM.call`.
