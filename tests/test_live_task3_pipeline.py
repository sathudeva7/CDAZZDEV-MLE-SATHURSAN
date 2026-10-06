"""Live check of the Task 3B pipeline: one full two-agent run, then a cache hit.

Skipped by default (see pytest.ini); run with: pytest -m live
Skipped, with the reason, when the primary LLM key is not set. It spends
about 30K tokens of the free tier's 200K a day and takes 2-3 minutes,
mostly SDK retries on the 8K-tokens-a-minute limit. Logs and the cache go to
a temporary folder.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 3: the 3B two-agent pipeline with the critique loop and the persistent cache, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import pytest

from common.llm_config import active_profile, api_key
from task1_financial.pipeline import DEFAULT_TICKER
from task3_agentic.pipeline import ResearchPipeline
from task3_agentic.tools import ToolSession


@pytest.mark.live
@pytest.mark.skipif(not api_key(active_profile().primary), reason=f"{active_profile().primary.api_key_env} is not set")
def test_live_pipeline_runs_the_critique_loop_then_hits_the_cache(tmp_path):
    pipeline = ResearchPipeline(ToolSession(subject=DEFAULT_TICKER, log_dir=tmp_path), cache_dir=tmp_path / "cache")

    first = pipeline.run(DEFAULT_TICKER)
    second = pipeline.run(DEFAULT_TICKER)

    assert first.report.generated_by == "llm", first.report.warnings
    assert first.brief.price is not None and first.brief.generated_by == "llm"
    assert first.request.needs in ("price_data", "volatility", "sentiment")
    assert first.report.clarification_used
    assert first.cache_path.exists()
    assert second.cached and second.report == first.report
