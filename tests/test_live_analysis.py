"""Live checks of the two model providers behind Task 1B: one Jev request and one LLM Recommendation.

Skipped by default (see pytest.ini); run with: pytest -m live
Each test is also skipped, with the reason, when its key is not set, so a run
without keys still passes. Logs go to a temporary folder, not the submitted log.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import pytest

from common.jev import JEV_API_KEY_ENV, PROBABILITY_SUM_TOLERANCE, JevClient
from common.llm import StructuredLLM
from common.llm_config import active_profile, api_key, secret
from task1_financial.prompts import HEADLINE_SENTIMENT_QUESTION
from task1_financial.recommendation import recommend
from task1_financial.schemas import SentimentSummary
from tests.test_recommendation import SUMMARY, indicator_frame


@pytest.mark.live
@pytest.mark.skipif(not secret(JEV_API_KEY_ENV), reason=f"{JEV_API_KEY_ENV} is not set")
def test_live_jev_labels_a_clear_headline(tmp_path):
    variables = {"company": "Apple Inc.", "ticker": "AAPL", "publisher": "Reuters", "headline": "Apple beats quarterly revenue estimates on record iPhone sales"}

    result = JevClient(tmp_path).choose(HEADLINE_SENTIMENT_QUESTION, variables)

    assert result.ok, result.error
    assert result.value.choice == "positive"
    assert sum(result.value.probabilities.values()) == pytest.approx(1.0, abs=PROBABILITY_SUM_TOLERANCE)


@pytest.mark.live
@pytest.mark.skipif(not api_key(active_profile().primary), reason=f"{active_profile().primary.api_key_env} is not set")
def test_live_llm_gives_a_valid_recommendation(tmp_path):
    sentiment = SentimentSummary(score=0.1, label="neutral", scored=15, by_jev=15, by_llm=0, failed=0, counts={"positive": 5, "negative": 3, "neutral": 7})

    result = recommend(indicator_frame(), SUMMARY, sentiment, StructuredLLM(tmp_path))

    assert result.ok, "every provider failed; see the warnings in the test log"
    assert result.value.recommendation in ("Buy", "Hold", "Sell")
