"""agent_trace.jsonl: one JSON line per tool call, handoff, critique and cache lookup.

The brief asks for every tool call's name, inputs, output (cut to 200
characters) and wall-clock duration. Each `tool_call` line carries:

    ts, run_id, event, agent, tool, args, why, status, output, output_chars,
    duration_ms, cache_hit

`output` is the digest the agent read (ToolResult.for_llm), cut to
TRACE_OUTPUT_CHARS; `output_chars` is its full length, so a reader can tell
when it was cut. `why` is the agent's stated reason (None for calls made by
code). Other events (agent_turn, report, handoff, critique, cache) share ts,
run_id and event, plus their own fields. The tool session writes every tool_call line
itself, so no tool can skip the trace.

A lock serialises writes, because LangGraph runs parallel tool calls in threads.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 1: the five agent tools, their tests and the new-tool skill, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

TRACE_OUTPUT_CHARS = 200
TRACE_FILE_NAME = "agent_trace.jsonl"
DEFAULT_LOG_DIR = Path(__file__).parent / "logs"

# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07
# agent_turn records each model reply (which tools it asked for, which model answered,
# tokens), so the trace shows the decisions between tool calls, not only the calls.
TraceEvent = Literal["tool_call", "agent_turn", "report", "handoff", "critique", "cache"]


class TraceLog:
    """Appends trace events for one run to <log_dir>/agent_trace.jsonl."""

    def __init__(self, run_id: str, log_dir: Path | str = DEFAULT_LOG_DIR) -> None:
        self.run_id = run_id
        self.path = Path(log_dir) / TRACE_FILE_NAME
        self._lock = threading.Lock()

    def tool_call(
        self,
        *,
        agent: str,
        tool: str,
        args: dict[str, Any],
        status: str,
        output: str,
        duration_ms: float,
        cache_hit: bool,
        why: str | None = None,
    ) -> None:
        self.write(
            "tool_call",
            agent=agent,
            tool=tool,
            args=args,
            why=why,
            status=status,
            output=output[:TRACE_OUTPUT_CHARS],
            output_chars=len(output),
            duration_ms=round(duration_ms, 1),
            cache_hit=cache_hit,
        )

    def write(self, event: TraceEvent, **fields: Any) -> None:
        line = {"ts": datetime.now(timezone.utc).isoformat(), "run_id": self.run_id, "event": event, **fields}
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, default=str) + "\n")
