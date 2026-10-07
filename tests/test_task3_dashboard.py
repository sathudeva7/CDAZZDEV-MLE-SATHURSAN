"""Offline tests for the trace dashboard: the views on a small hand-written trace, and the app headless.

The app test uses Streamlit's AppTest and is skipped when Streamlit is not
installed (it lives in task3_agentic/dashboard/requirements.txt only).
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Streamlit trace dashboard as settled in the grilling round (go with recommendations)', Date: 2026-10-07

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from task3_agentic.dashboard import views


def call(run_id, ts, agent, tool, status="ok", ms=1000.0, output="{}", **args):
    return {"ts": ts, "run_id": run_id, "event": "tool_call", "agent": agent, "tool": tool, "args": args or {"ticker": "AAPL"},
            "why": f"why {tool}", "status": status, "output": output, "output_chars": len(output), "duration_ms": ms, "cache_hit": False}


def turn(run_id, ts, agent, number, tools, tokens=100):
    return {"ts": ts, "run_id": run_id, "event": "agent_turn", "agent": agent, "turn": number, "model": "m", "tool_calls": tools, "text": "", "tokens": tokens}


EVENTS = [
    call("direct1", "2026-10-07T08:00:01+00:00", "direct", "get_price_data"),
    turn("agent1", "2026-10-07T08:01:00+00:00", "single", 1, ["get_news"]),
    call("agent1", "2026-10-07T08:01:02+00:00", "single", "get_news", status="error", ms=500.0, output='{"error": "injected failure: get_news is in failing_tools"}'),
    turn("agent1", "2026-10-07T08:01:05+00:00", "single", 2, ["web_search"], tokens=200),
    call("agent1", "2026-10-07T08:01:09+00:00", "single", "web_search", ms=4000.0, query="AAPL news"),
    {"ts": "2026-10-07T08:01:10+00:00", "run_id": "agent1", "event": "report", "agent": "single", "generated_by": "llm", "risks": ["a", "b", "c"], "hedge": "collar", "warnings": []},
    {"ts": "2026-10-07T08:02:00+00:00", "run_id": "pipe1", "event": "cache", "action": "miss", "path": "/x/cache/AAPL_2026-10-07.json", "reason": "no file"},
    call("pipe1", "2026-10-07T08:02:03+00:00", "A", "get_price_data"),
    {"ts": "2026-10-07T08:02:04+00:00", "run_id": "pipe1", "event": "handoff", "sender": "A", "receiver": "B", "schema": "DataBrief", "content": {"ticker": "AAPL"}},
    {"ts": "2026-10-07T08:02:05+00:00", "run_id": "pipe1", "event": "critique", "step": "request", "sender": "B", "receiver": "A", "content": {"needs": "volatility"}},
    {"ts": "2026-10-07T08:03:00+00:00", "run_id": "hit1", "event": "cache", "action": "hit", "path": "/x/cache/AAPL_2026-10-07.json"},
]


def test_each_run_is_labelled_from_its_events():
    table = views.runs_table(EVENTS)

    assert list(table["label"]) == ["Direct tool calls", "3A research agent (get_news failing)", "3B pipeline", "3B cache hit"]
    assert list(table["ticker"]) == ["AAPL", "AAPL", "AAPL", "AAPL"]  # the cache hit's ticker comes from its file name
    assert list(table["problems"]) == [0, 1, 0, 0]


def test_summary_counts_calls_problems_time_and_tokens():
    run = views.group_runs(EVENTS)["agent1"]

    assert views.summary(run) == {"tool_calls": 2, "problems": 1, "tool_seconds": 4.5, "model_turns": 2, "tokens": 300, "cache_hits": 0}


def test_timeline_starts_each_call_at_its_end_minus_its_duration():
    frame = views.timeline(views.group_runs(EVENTS)["agent1"])

    search = frame[frame["tool"] == "web_search"].iloc[0]
    assert search["end"] - search["start"] == pd.Timedelta(seconds=4)
    assert str(search["start"]) == "2026-10-07 08:01:05+00:00"


def test_messages_and_lanes_cover_both_agents():
    run = views.group_runs(EVENTS)["pipe1"]
    notes = views.messages(run)

    assert list(notes["message"]) == ["A -> B: DataBrief", "B -> A: ClarificationRequest"]
    assert views.lanes(notes) == ["A", "B"]


def test_steps_give_one_titled_step_per_event():
    titles = [step["title"] for step in views.steps(views.group_runs(EVENTS)["agent1"])]

    assert titles == [
        "[single] turn 1: get_news",
        "[single] get_news -> error (500 ms)",
        "[single] turn 2: web_search",
        "[single] web_search -> ok (4,000 ms)",
        "[single] report written by llm, hedge collar",
    ]


def test_tool_stats_per_tool():
    stats = views.tool_stats(views.group_runs(EVENTS)["agent1"]).set_index("tool")

    assert stats.loc["get_news", "ok"] == 0 and stats.loc["web_search", "mean_ms"] == 4000.0
    assert views.tool_stats(views.group_runs(EVENTS)["hit1"]).empty


def test_parse_lines_reads_uploaded_bytes():
    raw = (json.dumps(EVENTS[0]) + "\n\n" + json.dumps(EVENTS[1]) + "\n").encode().splitlines()

    assert views.parse_lines(raw) == EVENTS[:2]


def test_app_renders_every_run_without_an_exception(tmp_path, monkeypatch):
    testing = pytest.importorskip("streamlit.testing.v1")
    trace = tmp_path / "agent_trace.jsonl"
    trace.write_text("".join(json.dumps(event) + "\n" for event in EVENTS))
    monkeypatch.setenv("AGENT_TRACE_PATH", str(trace))
    app = Path(__file__).parents[1] / "task3_agentic" / "dashboard" / "app.py"

    at = testing.AppTest.from_file(str(app), default_timeout=30).run()
    assert not at.exception
    assert at.subheader[0].value.startswith("3B pipeline")  # opens on the 3B run
    for run_id in ("direct1", "agent1", "hit1"):
        at.sidebar.radio[0].set_value(run_id).run()
        assert not at.exception, run_id


def test_timeline_rows_group_tools_under_their_agent_then_messages():
    run = views.group_runs(EVENTS)["pipe1"]

    assert views.row_order(views.timeline(run), views.messages(run)) == ["A · get_price_data", views.MESSAGES_ROW]
