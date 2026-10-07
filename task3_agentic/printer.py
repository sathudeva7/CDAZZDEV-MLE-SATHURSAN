"""Prints an agent's run as it happens, for the notebook's visible message trace.

Pass `print_update` as `on_update` to ResearchAgent.research or ask. Each
LangGraph step prints as it arrives:

    [single] turn 2 (openai/gpt-oss-120b, 3,104 tokens)
      -> web_search(query='AAPL analyst downgrade risks')
         why: get_news failed (injected); the hint suggests web_search for recent news
    [single] tools
      x get_news error: injected failure: get_news is in failing_tools ...
         hint: Alternatives: web_search: search 'AAPL stock news this week' ...
      ok web_search: {"query": "AAPL analyst downgrade risks", "backend": "text", ...
    [single] report written by llm: 3 risks, hedge protective_put, 0 warnings

A tool call line shows the model's `why`, so each observe -> decide step is
visible; the tool result lines show what the model observed next.

The 3B pipeline adds its own steps: the news intake, each handoff printed in
full (the DataBrief, the ClarificationRequest, the ClarificationResponse),
and every cache lookup and save.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add a stronger paid OpenAI model (gpt-6.1-sol) for testing, in a separate file so it can be deleted before submission', Date: 2026-10-07

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from itertools import pairwise
from typing import Any

from langchain_core.messages import AIMessage

from task3_agentic.agent import message_text
from task3_agentic.tools import WHY_ARG

DIGEST_PREVIEW_CHARS = 160
ANSWER_PREVIEW_CHARS = 600


def print_update(agent: str, step: dict[str, Any], write: Callable[[str], None] = print) -> None:
    """Print one LangGraph `updates` step: {node name: what that node returned}."""
    for node, update in step.items():
        if not update:
            continue
        if node == "agent":
            _print_turn(agent, update, write)
        elif node == "tools":
            _print_tools(agent, update, write)
        elif node == "finish":
            _print_finish(agent, update, write)
        elif node in PIPELINE_PRINTERS:
            PIPELINE_PRINTERS[node](update, write)


def _print_turn(agent: str, update: dict, write: Callable[[str], None]) -> None:
    for message in update.get("messages", []):
        if not isinstance(message, AIMessage):
            continue
        model = message.response_metadata.get("model_name") or "no model"
        tokens = (message.usage_metadata or {}).get("total_tokens")
        write(f"[{agent}] turn {update.get('turns')} ({model}{f', {tokens:,} tokens' if tokens else ''})")
        for call in message.tool_calls:
            args = {key: value for key, value in call["args"].items() if key != WHY_ARG}
            shown = ", ".join(f"{key}={_short(value)}" for key, value in args.items())
            write(f"  -> {call['name']}({shown})")
            if call["args"].get(WHY_ARG):
                write(f"     why: {call['args'][WHY_ARG]}")
        text = message_text(message)  # a string, or content blocks on OpenAI's Responses API
        if text and not message.tool_calls:
            write(f"  says: {text[:ANSWER_PREVIEW_CHARS]}")
    for warning in update.get("warnings", []):
        write(f"  ! {warning}")


def _print_tools(agent: str, update: dict, write: Callable[[str], None]) -> None:
    write(f"[{agent}] tools")
    for obs in update.get("observations", []):
        digest = json.loads(obs["digest"])
        if obs["status"] == "ok":
            data = json.dumps(digest.get("data"), default=str)
            write(f"  ok {obs['tool']}: {data[:DIGEST_PREVIEW_CHARS]}{'...' if len(data) > DIGEST_PREVIEW_CHARS else ''}")
        else:
            problem = digest.get("error") or "; ".join(digest.get("warnings", [])) or "no detail"
            write(f"  x {obs['tool']} {obs['status']}: {problem}")
            if digest.get("hint"):
                write(f"     hint: {digest['hint']}")


def _print_finish(agent: str, update: dict, write: Callable[[str], None]) -> None:
    result = update.get("result") or {}
    if "top_risks" in result:
        write(
            f"[{agent}] report written by {result['generated_by']}: {len(result['top_risks'])} risks, "
            f"hedge {result['hedge']['strategy']}, {len(result.get('warnings', []))} warnings"
        )
        if result.get("clarification_used"):
            write(f"  clarification used: {result['clarification_used']}")
    # Other finish steps are 3B handoffs, which the pipeline prints in full next.


def _short(value: Any) -> str:
    """A compact repr for an argument: long lists of headlines show their length only."""
    if isinstance(value, list) and len(value) > 2:
        return f"[{len(value)} items]"
    return repr(value)


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 3: the 3B two-agent pipeline with the critique loop and the persistent cache, as designed in the grilling rounds', Date: 2026-10-07
# --- the 3B pipeline's own steps ----------------------------------------------------------


def _print_cache(update: dict, write: Callable[[str], None]) -> None:
    action = update["action"]
    if action == "hit":
        write(f"[pipeline] cache hit: {update['path']} (saved {update['saved_at']}); loading it instead of running the agents")
    elif action == "miss":
        write(f"[pipeline] cache miss: no valid {update['path']}; running both agents")
    elif action == "skip":
        write("[pipeline] cache skipped: force_refresh=True")
    elif action == "skip_save":
        write(f"[pipeline] not cached: {update['reason']}")
    else:
        write(f"[pipeline] saved the research brief to {update['path']}")


def _print_intake(update: dict, write: Callable[[str], None]) -> None:
    write(f"[pipeline] news intake for Agent A ({update['status']}): {len(update['headlines'])} headlines")
    for title in update["headlines"]:
        write(f"  - {title}")


def _print_handoff(brief: dict, write: Callable[[str], None]) -> None:
    write("[A -> B] handoff: DataBrief")
    for key in ("price", "volatility", "sentiment"):
        write(f"  {key}: {json.dumps(_rounded(brief[key]), default=str)}")
    levels = brief.get("hedge_levels")
    if levels:
        candidates = ", ".join(f"{c['name']} {c['level']}" for c in levels["candidates"])
        write(f"  hedge_levels: expected move ±{levels['expected_move']} ({levels['expected_move_pct']}%) from {levels['price']}; {candidates}")
    else:
        write("  hedge_levels: null")
    write(f"  key_observations ({brief['generated_by']}):")
    for item in brief["key_observations"]:
        write(f"    - {item}")
    write(f"  data_gaps: {brief['data_gaps']}")


def _print_request(request: dict, write: Callable[[str], None]) -> None:
    params = {key: request[key] for key in ("period", "window", "headlines") if request.get(key)}
    write(f"[B -> A] clarification request ({request['needs']}{f', {params}' if params else ''})")
    write(f"  question: {request['question']}")
    write(f"  why: {request['why']}")


def _print_clarification(response: dict, write: Callable[[str], None]) -> None:
    write(f"[A -> B] clarification answer (fulfilled by {response['fulfilled_by']})")
    write(f"  answer: {response['answer']}")
    for key in ("price", "volatility", "sentiment"):
        if response.get(key):
            write(f"  {key}: {json.dumps(_rounded(response[key]), default=str)}")


def _rounded(value: Any) -> Any:
    """`value` with every float rounded to 2 decimals, for readable printing."""
    if isinstance(value, float):
        return round(value, 2)
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


PIPELINE_PRINTERS: dict[str, Callable[[dict, Callable[[str], None]], None]] = {
    "cache": _print_cache,
    "news_intake": _print_intake,
    "handoff": _print_handoff,
    "critique_request": _print_request,
    "clarification": _print_clarification,
}


# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Start PR 4: the Task 3 notebook, laid out as settled in the grilling rounds', Date: 2026-10-07
# --- observe -> replan, read back from the trace -------------------------------------------


def replan_cycles(events: Sequence[dict], agent: str) -> list[dict]:
    """Each model turn that chose tools after an earlier turn's results: what it had observed, then what it chose.

    Built from the trace, where each `agent_turn` line is followed by the
    `tool_call` lines it asked for. Calls within one turn were chosen together,
    so a cycle pairs one turn's results with the next turn's choices.
    """
    turns: list[dict] = []
    for event in events:
        if event.get("agent") != agent:
            continue
        if event["event"] == "agent_turn":
            turns.append({"turn": event["turn"], "calls": []})
        elif event["event"] == "tool_call" and turns:
            turns[-1]["calls"].append(event)
    return [
        {"turn": current["turn"], "observed": previous["calls"], "chose": current["calls"]}
        for previous, current in pairwise(turns)
        if previous["calls"] and current["calls"]
    ]


def print_replan_cycles(events: Sequence[dict], agent: str, write: Callable[[str], None] = print) -> None:
    cycles = replan_cycles(events, agent)
    if not cycles:
        write(f"[{agent}] no observe -> replan cycle: the model chose all its tools in one turn")
    for cycle in cycles:
        write(f"cycle into turn {cycle['turn']}")
        for call in cycle["observed"]:
            write(f"  observed {call['tool']} -> {call['status']}: {call['output'][:DIGEST_PREVIEW_CHARS]}")
        for call in cycle["chose"]:
            shown = ", ".join(f"{key}={_short(value)}" for key, value in call["args"].items())
            write(f"  chose    {call['tool']}({shown})")
            write(f"           why: {call['why']}")
