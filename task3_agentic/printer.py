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
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.messages import AIMessage

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
        text = message.content if isinstance(message.content, str) else ""
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
    else:
        write(f"[{agent}] finished")


def _short(value: Any) -> str:
    """A compact repr for an argument: long lists of headlines show their length only."""
    if isinstance(value, list) and len(value) > 2:
        return f"[{len(value)} items]"
    return repr(value)
