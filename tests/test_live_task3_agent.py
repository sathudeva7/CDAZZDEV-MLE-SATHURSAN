"""Live check of the Task 3A agent: one research run on the default ticker and one follow-up.

Skipped by default (see pytest.ini); run with: pytest -m live
Skipped, with the reason, when the primary LLM key is not set. It spends
about 25K tokens of the free tier's 200K a day. Logs go to a temporary folder.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import pytest

from common.llm_config import active_profile, api_key
from task1_financial.pipeline import DEFAULT_TICKER
from task3_agentic.agent import ResearchAgent
from task3_agentic.tools import ToolSession

MIN_TOOLS_USED = 3  # the report needs at least prices, volatility or sentiment, and news or commentary


@pytest.mark.live
@pytest.mark.skipif(not api_key(active_profile().primary), reason=f"{active_profile().primary.api_key_env} is not set")
def test_live_research_run_then_follow_up_from_memory(tmp_path):
    agent = ResearchAgent(ToolSession(subject=DEFAULT_TICKER, log_dir=tmp_path))

    run = agent.research(DEFAULT_TICKER)
    follow_up = agent.ask("What 30-day volatility did you find?", thread_id=run.thread_id)

    report = run.report
    assert report.generated_by == "llm", report.warnings
    assert len(report.top_risks) == 3 and all(risk.evidence for risk in report.top_risks)
    assert all(leg.level is not None for leg in report.hedge.legs if leg.instrument != "shares")
    assert len(report.tools_called) >= MIN_TOOLS_USED
    assert follow_up.tool_calls == 0 and follow_up.answer
