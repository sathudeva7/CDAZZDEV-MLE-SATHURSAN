"""Task 3 trace dashboard: browse agent_trace.jsonl run by run.

Run from the repo root:
    pip install -r task3_agentic/dashboard/requirements.txt
    streamlit run task3_agentic/dashboard/app.py

It opens the committed task3_agentic/logs/agent_trace.jsonl (or the file
named by AGENT_TRACE_PATH); the sidebar can load any other trace. All tables
come from views.py; this file only lays them out.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Streamlit trace dashboard as settled in the grilling round (go with recommendations)', Date: 2026-10-07

from __future__ import annotations

import os
import sys
from pathlib import Path

import altair as alt
import streamlit as st

# `streamlit run` puts this file's folder on sys.path, not the repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from task3_agentic.dashboard import views
from task3_agentic.trace import (
    DEFAULT_LOG_DIR,
    TRACE_FILE_NAME,
    read_trace,
)

TRACE_PATH_ENV = "AGENT_TRACE_PATH"
DEFAULT_TRACE = Path(os.environ.get(TRACE_PATH_ENV, DEFAULT_LOG_DIR / TRACE_FILE_NAME))
STATUS_COLOURS = {"ok": "#2e7d32", "empty": "#f9a825", "error": "#c62828", "refused": "#6a1b9a", "skipped": "#757575", "invalid_args": "#ef6c00"}
ROW_STEP_PX = 34  # height of one timeline row
UTC = alt.Scale(type="utc")  # the trace is in UTC; without this the browser shows its local time

st.set_page_config(page_title="Agent trace", layout="wide")
st.title("Task 3 agent trace")

# --- which trace, which run ---------------------------------------------------------------
uploaded = st.sidebar.file_uploader("Load another agent_trace.jsonl", type=["jsonl"])
if uploaded is not None:
    events, source = views.parse_lines(uploaded.getvalue().splitlines()), uploaded.name
else:
    events, source = read_trace(DEFAULT_TRACE), str(DEFAULT_TRACE.relative_to(REPO_ROOT) if DEFAULT_TRACE.is_relative_to(REPO_ROOT) else DEFAULT_TRACE)
if not events:
    st.info(f"No trace events in {source}. Run task3_agentic/task3_agentic_research.ipynb first, or load a file in the sidebar.")
    st.stop()

runs = views.group_runs(events)
table = views.runs_table(events)
st.sidebar.caption(f"{len(events)} events, {len(runs)} runs from {source}")
choices = {
    row.run_id: f"{row.label} · {row.ticker or '?'} · {row.started:%H:%M:%S} · {row.tool_calls} calls"
    for row in table.itertuples()
}
default = next((i for i, row in enumerate(table.itertuples()) if row.label == "3B pipeline"), 0)
run_id = st.sidebar.radio("Run", list(choices), index=default, format_func=choices.get)
run = runs[run_id]
st.subheader(choices[run_id])

with st.sidebar.expander("All runs"):
    st.dataframe(table, hide_index=True)

# --- headline numbers ----------------------------------------------------------------------
numbers = views.summary(run)
for column, (label, key) in zip(
    st.columns(6),
    [("Tool calls", "tool_calls"), ("Not ok", "problems"), ("Tool time (s)", "tool_seconds"),
     ("Model turns", "model_turns"), ("Tokens", "tokens"), ("Cache hits", "cache_hits")],
    strict=True,
):
    column.metric(label, f"{numbers[key]:,}")

timeline_tab, steps_tab, tools_tab = st.tabs(["Timeline", "Steps", "Tools"])

# --- timeline: one lane per agent, one bar per tool call ------------------------------------
with timeline_tab:
    calls = views.timeline(run)
    notes = views.messages(run)
    if calls.empty and notes.empty:
        st.write("This run made no tool calls (a cache hit loads the saved report instead).")
    else:
        order = views.row_order(calls, notes)
        y = alt.Y("row:N", sort=order, title=None, axis=alt.Axis(labelLimit=260))
        x = alt.X("start:T", title="time (UTC)", scale=UTC, axis=alt.Axis(format="%H:%M:%S"))
        status = alt.Color("status:N", scale=alt.Scale(domain=list(STATUS_COLOURS), range=list(STATUS_COLOURS.values())), legend=alt.Legend(orient="bottom"))
        tooltip = ["call", "agent", "tool", "status", alt.Tooltip("duration_ms:Q", format=",.0f"), "why"]
        layers = [
            alt.Chart(calls).mark_bar(height=16, cornerRadius=3).encode(x=x, x2="end:T", y=y, color=status, tooltip=tooltip),
            # A tick at each start keeps near-instant calls (cache hits) visible.
            alt.Chart(calls).mark_tick(thickness=3, size=20).encode(x=x, y=y, color=status, tooltip=tooltip),
        ]
        if not notes.empty:
            at = alt.X("at:T", scale=UTC, axis=alt.Axis(format="%H:%M:%S"))
            layers += [
                alt.Chart(notes).mark_point(shape="diamond", size=160, filled=True, color="#1565c0").encode(x=at, y=y, tooltip=["message", "at:T"]),
                alt.Chart(notes).mark_text(dy=-14, fontSize=11, color="#1565c0").encode(x=at, y=y, text="short:N"),
            ]
        st.altair_chart(alt.layer(*layers).properties(height=alt.Step(ROW_STEP_PX)), width="stretch")
        st.caption("One row per agent and tool. Bars run from each call's start to its end, coloured by status; hover for the agent's reason. Diamonds are messages between agents.")

# --- steps: the run in order -----------------------------------------------------------------
with steps_tab:
    for number, step in enumerate(views.steps(run), 1):
        with st.expander(f"{number}. {step['time']:%H:%M:%S} · {step['title']}", expanded=step["kind"] in ("handoff", "critique")):
            st.json(step["detail"], expanded=True)

# --- tools: totals per tool --------------------------------------------------------------------
with tools_tab:
    stats = views.tool_stats(run)
    if stats.empty:
        st.write("No tool calls in this run.")
    else:
        st.altair_chart(
            alt.Chart(stats).mark_bar().encode(
                x=alt.X("mean_ms:Q", title="mean duration (ms)"), y=alt.Y("tool:N", sort="-x", title=None),
                tooltip=["tool", "calls", "ok", "mean_ms", "max_ms"],
            ),
            width="stretch",
        )
        st.dataframe(stats, hide_index=True)
