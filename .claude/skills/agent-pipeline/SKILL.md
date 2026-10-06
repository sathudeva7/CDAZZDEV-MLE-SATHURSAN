---
name: agent-pipeline
description: Build, change or debug a Task 3 agent in task3_agentic/agent.py (the LangGraph loop, budgets, a finish step, the report, follow-up memory) or the 3B two-agent pipeline built on it. Use when touching agent.py, report.py, printer.py or prompts.py, or when an agent's run, trace or report looks wrong.
---

# Agent pipeline

Task 3 scores the loop's behaviour from the notebook output: autonomous tool
order (10), a visible observe-and-replan cycle (8), the report (10), error
handling (7), and in 3B tool restriction, handoff, trace and the critique
loop. The loop's shape, budgets and state rules live in the docstring of
`task3_agentic/agent.py`, and the report's hedge maths and checks in
`task3_agentic/report.py`. Read the relevant one first and keep it true in
the same change.

## Adding an agent

1. **Tools**: pass `tool_names` to `build_agent_graph`. Binding is the
   restriction, and the tools node refuses anything else, so prompt wording
   can't widen it.
2. **Prompt**: a `Prompt` constant in `task3_agentic/prompts.py`. The system
   text says what evidence the agent needs and when to stop. Tool order is
   decided by the model from what it observes.
3. **Finish step**: `finish(state) -> {"result": <JSON dict>, "warnings": [...]}`.
   It reads `state["observations"]`. Rebuild typed models with
   `Model.model_validate(obs["result"]["data"])`, because state holds only
   JSON: the in-memory checkpointer turns other objects into dicts. Any LLM
   call follows the `llm-structured-call` skill, with a fallback the run can end on.
4. **Trace**: write one event for the finish step
   (`session.trace.write("report" | "handoff" | ..., ...)`), so the Streamlit
   viewer and a reviewer see what was produced.
5. **Tests** in `tests/test_task3_agent.py`, with `ScriptedChat` and
   `tool_turn` from `tests/fakes.py`:
   - Script the model's turns, and build a new `AIMessage` per turn
     (`add_messages` replaces a message whose id it has seen).
   - Cover a failure the agent must route around, a budget limit, a refused
     tool, and every model failing.
   - Mutation-check each routing rule you add: break it, see red, restore.

## Checking a live run

Run it with `on_update=print_update`, then read in this order:

1. The `why` lines. Each one should cite something observed. Generic reasons
   point to a prompt problem.
2. The `x ... hint:` lines. The next turn should try the hinted alternative.
3. The report against the observations. Every number must appear in a
   digest, and the volatility reading must match its percentile.
4. The call log (`llm_calls.jsonl`) for repair retries and SDK retries
   (429s), and `agent_turn` tokens in the trace against the 8K-a-minute limit.

A prompt change keeps its `version` until the prompt has shipped in a merged
PR; after that, bump it.

Done when `pytest -q` passes, a live run on the free profile shows the
replan and a checked report, and the docstrings match the code.
