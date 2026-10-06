---
name: new-tool
description: Add or change a Task 3 agent tool in task3_agentic/tools.py (get_price_data, calculate_volatility, get_news, llm_sentiment, web_search, or a new one), its schema, digest, failure hint or trace line. Use when touching tools.py or task3_agentic/schemas.py, or when an agent needs data no tool returns.
---

# Agent tools

Tools earn 15 marks for "callable and returning correct data types" and carry
the 7-mark error-handling criterion: the agent can only replan around a
failure the tool reports clearly. The call contract (never raises, traced,
shared price cache, injected demo failures, hints filtered to the caller's
tools) lives in the docstring of `task3_agentic/tools.py`. Read it first, and
keep it true in the same change.

## Adding a tool

1. **Data model** in `task3_agentic/schemas.py`, with a `digest()` method.
   - The model keeps everything a caller could need, such as every bar in the period.
   - `digest()` is what the agent reads: the few numbers that drive a
     decision, rounded with `_round`. Every tool message is re-sent on each
     agent turn, and Groq's free tier allows 8K tokens a minute, so aim for
     under 500 tokens (measure `len(result.for_llm()) / 4` on a live call).
2. **Body** in `ToolSession`: `_my_tool(self, ...) -> tuple[ToolResult, bool]`.
   The bool is `cache_hit`.
   - Reuse Task 1's module for the data (`data`, `news`, `indicators`,
     `sentiment`) so each number has one source.
   - Bad arguments are clamped or replaced with the default, and `_warn` records it.
   - The source answering with nothing is status `empty`. The tool itself
     failing is status `error`. Any other exception is caught by `call()`.
   - An LLM call follows the `llm-structured-call` skill: a prompt in
     `task3_agentic/prompts.py`, `self._get_llm().call(...)` with a fallback,
     and a branch on `ok`.
3. **Register** it in every table, so the agent, the hint and the trace see it:
   the name constant and `TOOL_NAMES`, `self._tools`, `TOOL_DESCRIPTIONS`
   (written for the agent's model: what it returns and when to use it),
   `TOOL_ARGS` (plain `str`/`int` fields with descriptions, so out-of-range
   values reach the tool and get traced), `TOOL_ADVICE` and `ALTERNATIVES`.
   Add a public method that goes through `self.call(...)`.
4. **Wire alternatives both ways.** Add the new tool as an alternative under
   every existing tool whose gap it can fill.
5. **Tests** in `tests/test_task3_tools.py`, offline with the fakes there:
   - The ok path, checked against Task 1's own function or a hand-worked example.
   - Each `empty` and `error` path, with the hint naming the right alternative.
   - The hint for a restricted agent's tool list (see `AGENT_A_TOOLS`).
   - A mutation check: plant the bug the test guards against, confirm it goes red, restore.
   - One assertion in `tests/test_live_task3_tools.py` for the real source.

Done when `pytest -q` passes, a live run shows the digest size, and
`test_every_tool_has_a_description_and_argument_schema` covers the new name.

## Changing a tool

- Changing a digest or a description changes what the agent sees. Rerun the
  agent tests, and say in the PR which agent decision it could affect.
- Agent tool access is set where `langchain_tools(names, agent=...)` is
  called. The 3B restriction (A: price, volatility, sentiment; B: web_search,
  news) is a rubric row, so a change to who gets which tool needs the user's yes.
