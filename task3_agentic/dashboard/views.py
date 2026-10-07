"""Tables for the trace dashboard, built from agent_trace.jsonl events with pandas only.

No Streamlit import here, so every view is tested offline (tests/test_task3_dashboard.py)
and app.py stays a thin display layer.

A trace holds many runs, each named only by its run_id, so `run_label` infers
what a run was from its events: direct tool calls, the 3A agent, the 3B
pipeline or a 3B cache hit. A tool_call's `ts` is written when the call
ends, so its start is ts - duration_ms.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Streamlit trace dashboard as settled in the grilling round (go with recommendations)', Date: 2026-10-07

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd

INJECTED_FAILURE_MARK = "injected failure"
TEXT_PREVIEW_CHARS = 160
MS_PER_SECOND = 1000.0

# Agent names in the trace, in the order their rows are drawn.
AGENT_LANES = ("direct", "single", "pipeline", "A", "B")
MESSAGES_ROW = "messages A <-> B"
MESSAGE_SHORT_NAMES = {"DataBrief": "brief", "ClarificationRequest": "request", "ClarificationResponse": "answer"}


def parse_lines(lines: Iterable[str | bytes]) -> list[dict[str, Any]]:
    """Events from JSONL lines, such as an uploaded file; blank lines are skipped."""
    events = []
    for line in lines:
        text = line.decode("utf-8") if isinstance(line, bytes) else line
        if text.strip():
            events.append(json.loads(text))
    return events


def group_runs(events: Sequence[dict]) -> dict[str, list[dict]]:
    """Each run's events in write order, keyed by run_id, in the order runs first appear."""
    runs: dict[str, list[dict]] = {}
    for event in events:
        runs.setdefault(event["run_id"], []).append(event)
    return runs


def run_label(run: Sequence[dict]) -> str:
    """What the run was, read from its events: 'Direct tool calls', '3A research agent', '3B pipeline' or '3B cache hit'."""
    kinds = {event["event"] for event in run}
    agents = {event.get("agent") for event in run if event["event"] == "tool_call"}
    if "handoff" in kinds:
        label = "3B pipeline"
    elif kinds == {"cache"}:
        label = "3B cache hit" if any(event.get("action") == "hit" for event in run) else "3B cache lookup"
    elif "single" in agents or "agent_turn" in kinds:
        label = "3A research agent"
    else:
        label = "Direct tool calls"
    failing = sorted({e["tool"] for e in run if e["event"] == "tool_call" and INJECTED_FAILURE_MARK in str(e.get("output", ""))})
    if failing:
        label += f" ({', '.join(failing)} failing)"
    return label


def run_ticker(run: Sequence[dict]) -> str | None:
    """The ticker the run was about: the first ticker argument, else the cache file's name."""
    for event in run:
        ticker = (event.get("args") or {}).get("ticker") if event["event"] == "tool_call" else None
        if ticker:
            return str(ticker).upper()
    for event in run:
        if event["event"] == "cache" and event.get("path"):
            return str(event["path"]).rsplit("/", 1)[-1].split("_", 1)[0]
    return None


def runs_table(events: Sequence[dict]) -> pd.DataFrame:
    """One row per run: id, label, ticker, start time, tool calls and problems."""
    rows = []
    for run_id, run in group_runs(events).items():
        calls = [e for e in run if e["event"] == "tool_call"]
        rows.append({
            "run_id": run_id,
            "label": run_label(run),
            "ticker": run_ticker(run),
            "started": pd.to_datetime(run[0]["ts"], utc=True),
            "tool_calls": len(calls),
            "problems": sum(e["status"] != "ok" for e in calls),
        })
    return pd.DataFrame(rows, columns=["run_id", "label", "ticker", "started", "tool_calls", "problems"])


def summary(run: Sequence[dict]) -> dict[str, Any]:
    """The headline numbers for one run."""
    calls = [e for e in run if e["event"] == "tool_call"]
    tokens = [e.get("tokens") or 0 for e in run if e["event"] == "agent_turn"]
    return {
        "tool_calls": len(calls),
        "problems": sum(e["status"] != "ok" for e in calls),
        "tool_seconds": round(sum(e["duration_ms"] for e in calls) / MS_PER_SECOND, 1),
        "model_turns": len(tokens),
        "tokens": sum(tokens),
        "cache_hits": sum(bool(e.get("cache_hit")) for e in calls) + sum(e.get("action") == "hit" for e in run if e["event"] == "cache"),
    }


def timeline(run: Sequence[dict]) -> pd.DataFrame:
    """One row per tool call with its start and end, for a Gantt chart with one lane per agent."""
    rows = []
    for number, event in enumerate((e for e in run if e["event"] == "tool_call"), 1):
        end = pd.to_datetime(event["ts"], utc=True)
        rows.append({
            "call": number,
            "agent": event["agent"],
            "tool": event["tool"],
            "row": f"{event['agent']} · {event['tool']}",
            "status": event["status"],
            "start": end - pd.Timedelta(milliseconds=event["duration_ms"]),
            "end": end,
            "duration_ms": event["duration_ms"],
            "why": event.get("why") or "",
        })
    return pd.DataFrame(rows, columns=["call", "agent", "tool", "row", "status", "start", "end", "duration_ms", "why"])


def messages(run: Sequence[dict]) -> pd.DataFrame:
    """The messages between agents (handoff and critique events), as points on the same time axis."""
    rows = [
        {
            "at": pd.to_datetime(e["ts"], utc=True),
            "agent": e["sender"],
            "row": MESSAGES_ROW,
            "message": f"{e['sender']} -> {e['receiver']}: {_message_name(e)}",
            "short": f"{e['sender']}->{e['receiver']} {MESSAGE_SHORT_NAMES.get(_message_name(e), _message_name(e))}",
        }
        for e in run
        if e["event"] in ("handoff", "critique")
    ]
    return pd.DataFrame(rows, columns=["at", "agent", "row", "message", "short"])


def lanes(frame: pd.DataFrame) -> list[str]:
    """The agents present, in drawing order."""
    present = set(frame["agent"]) if len(frame) else set()
    return [agent for agent in AGENT_LANES if agent in present] + sorted(present - set(AGENT_LANES))


def row_order(calls: pd.DataFrame, notes: pd.DataFrame) -> list[str]:
    """Timeline rows: grouped by agent in lane order, each agent's tools in first-call order, then the messages row."""
    order = []
    for agent in lanes(calls):
        order += list(dict.fromkeys(calls.loc[calls["agent"] == agent, "row"]))
    return order + ([MESSAGES_ROW] if len(notes) else [])


def steps(run: Sequence[dict]) -> list[dict[str, Any]]:
    """The run in order, one step per event: a one-line title and the detail to show when expanded."""
    return [_step(event) for event in run]


def tool_stats(run: Sequence[dict]) -> pd.DataFrame:
    """Per tool: calls, ok calls, mean and max duration."""
    calls = pd.DataFrame([e for e in run if e["event"] == "tool_call"])
    if calls.empty:
        return pd.DataFrame(columns=["tool", "calls", "ok", "mean_ms", "max_ms"])
    return (
        calls.assign(is_ok=calls["status"] == "ok")
        .groupby("tool")
        .agg(calls=("tool", "size"), ok=("is_ok", "sum"), mean_ms=("duration_ms", "mean"), max_ms=("duration_ms", "max"))
        .round(1)
        .reset_index()
    )


def _step(event: dict) -> dict[str, Any]:
    kind = event["event"]
    if kind == "tool_call":
        title = f"[{event['agent']}] {event['tool']} -> {event['status']} ({event['duration_ms']:,.0f} ms)"
        detail = {"args": event.get("args"), "why": event.get("why"), "output (first 200 chars)": event.get("output"), "cache_hit": event.get("cache_hit")}
    elif kind == "agent_turn":
        chose = ", ".join(event.get("tool_calls") or []) or "answers in text"
        title = f"[{event['agent']}] turn {event['turn']}: {chose}"
        detail = {"model": event.get("model"), "tokens": event.get("tokens"), "text": event.get("text")}
    elif kind in ("handoff", "critique"):
        title = f"{event['sender']} -> {event['receiver']}: {_message_name(event)}"
        detail = event.get("content")
    elif kind == "report":
        title = f"[{event['agent']}] report written by {event['generated_by']}, hedge {event['hedge']}"
        detail = {"risks": event.get("risks"), "warnings": event.get("warnings")}
    else:  # cache
        title = f"cache {event.get('action')}" + (f": {event['reason']}" if event.get("reason") else "")
        detail = {key: event.get(key) for key in ("path", "reason") if event.get(key)}
    return {"kind": kind, "time": pd.to_datetime(event["ts"], utc=True), "title": title, "detail": detail}


def _message_name(event: dict) -> str:
    if event["event"] == "handoff":
        return event.get("schema", "handoff")
    return "ClarificationRequest" if event.get("step") == "request" else "ClarificationResponse"
