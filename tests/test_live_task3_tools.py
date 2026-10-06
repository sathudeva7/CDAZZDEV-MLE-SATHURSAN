"""Live checks of Task 3's five tools against Yahoo, the news feeds, DuckDuckGo and the LLM.

Skipped by default (see pytest.ini); run with: pytest -m live
The LLM check is also skipped, with the reason, when its key is not set.
Trace and call logs go to a temporary folder, not the submitted log.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 1: the five agent tools, their tests and the new-tool skill, as designed in the grilling rounds', Date: 2026-10-07

from __future__ import annotations

import pytest

from common.llm_config import active_profile, api_key
from task1_financial.pipeline import DEFAULT_TICKER
from task3_agentic.tools import ToolSession


@pytest.mark.live
def test_live_data_tools_for_the_default_ticker(tmp_path):
    session = ToolSession(subject=DEFAULT_TICKER, log_dir=tmp_path)

    price = session.get_price_data(DEFAULT_TICKER, "3mo")
    vol = session.calculate_volatility(DEFAULT_TICKER)
    headlines = session.get_news(DEFAULT_TICKER, 5)
    search = session.web_search(f"{DEFAULT_TICKER} analyst price target")

    assert price.status == "ok" and price.data.latest.indicators["sma_200"] is not None
    assert vol.status == "ok" and 0 < vol.data.current_pct < 200
    assert headlines.status == "ok" and headlines.data.headlines
    assert search.status == "ok", search.warnings
    assert search.data.hits[0].title


@pytest.mark.live
@pytest.mark.skipif(not api_key(active_profile().primary), reason=f"{active_profile().primary.api_key_env} is not set")
def test_live_llm_sentiment_labels_clear_headlines(tmp_path):
    session = ToolSession(subject="Apple Inc. (AAPL)", log_dir=tmp_path)

    result = session.llm_sentiment(
        [
            "Apple beats quarterly revenue estimates on record iPhone sales",
            "Apple faces EU antitrust fine over App Store rules",
        ]
    )

    assert result.status == "ok", result.error
    assert [item.sentiment for item in result.data.items] == ["positive", "negative"]
