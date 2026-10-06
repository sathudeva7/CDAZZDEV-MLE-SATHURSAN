"""One live read of the real news feeds, to confirm the formats the offline tests assume.

Skipped by default (see pytest.ini). Run it with: pytest -m live
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1A step 2, news headlines, as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from task1_financial.news import MIN_HEADLINES, NEWS_MAX_AGE_DAYS, fetch_headlines
from task1_financial.pipeline import DEFAULT_TICKER, MARKET_TIMEZONE


@pytest.mark.live
def test_live_headlines_for_the_default_ticker():
    today = datetime.now(MARKET_TIMEZONE).date()

    result = fetch_headlines(DEFAULT_TICKER, today)

    # The brief's minimum. Yahoo may be rate-limiting (a warning), but Google should cover it.
    assert len(result.headlines) >= MIN_HEADLINES, result.warnings
    oldest_allowed = datetime.now(timezone.utc) - timedelta(days=NEWS_MAX_AGE_DAYS + 1)
    for headline in result.headlines:
        assert headline.title and headline.published >= oldest_allowed
    assert len({h.title.lower() for h in result.headlines}) == len(result.headlines)
