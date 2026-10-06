"""Offline tests for task1_financial/sentiment.py: who decides, what is sent, and the Sentiment score.

Jev and the LLM are scripted fakes (tests/fakes.py). The score cases are the
hand-checked examples from the design discussion.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from task1_financial.news import Headline
from task1_financial.schemas import (
    HeadlineReason,
    HeadlineSentiment,
    HeadlineSentimentAnswer,
)
from task1_financial.sentiment import (
    REASON_UNAVAILABLE,
    SENTIMENT_LABEL_THRESHOLD,
    score_headlines,
    summarise_sentiment,
)
from tests.fakes import ScriptedJev, ScriptedLLM

PUBLISHED = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
CEO_SELLS = Headline(title="Apple CEO sells $8.46 million in stock", publisher="Investing.com", published=PUBLISHED, url=None, feed="google")
BEATS = Headline(title="Apple beats revenue estimates", publisher=None, published=PUBLISHED, url=None, feed="yahoo")
NEGATIVE = ("negative", {"positive": 0.0, "neutral": 0.29, "negative": 0.71})
POSITIVE = ("positive", {"positive": 0.85, "neutral": 0.10, "negative": 0.05})


def score(headlines, jev_replies, llm_replies):
    jev, llm = ScriptedJev(list(jev_replies)), ScriptedLLM(list(llm_replies))
    return score_headlines(headlines, "AAPL", "Apple Inc.", jev, llm), jev, llm


# --- who decides ------------------------------------------------------------


def test_jev_decides_and_the_llm_explains():
    reason = HeadlineReason(brief_reason="Insider selling by the CEO may signal weaker confidence in near-term prospects.")
    [item], jev, llm = score([CEO_SELLS], [NEGATIVE], [reason])

    assert item.scored_by == "jev"
    assert (item.sentiment, item.confidence, item.score) == ("negative", 0.56, pytest.approx(-0.71))
    assert item.probabilities == NEGATIVE[1]
    assert item.headline == CEO_SELLS.title and item.brief_reason == reason.brief_reason
    # Jev judged the company, publisher and title; the LLM was told Jev's label and probabilities.
    assert jev.requests[0]["state"] == "Company: Apple Inc. (AAPL)\nPublisher: Investing.com\nHeadline: Apple CEO sells $8.46 million in stock"
    sent = llm.requests[0]
    assert sent["prompt"] == "headline_reason" and sent["schema"] is HeadlineReason
    assert sent["variables"]["sentiment"] == "negative"
    assert sent["variables"]["probabilities"] == "negative 71%, neutral 29%, positive 0%"


def test_jev_decision_is_kept_when_the_reason_fails():
    [item], _, _ = score([CEO_SELLS], [NEGATIVE], [None])

    assert item.scored_by == "jev" and item.sentiment == "negative"
    assert item.brief_reason == REASON_UNAVAILABLE


def test_llm_decides_when_jev_fails():
    answer = HeadlineSentimentAnswer(headline="Apple beats estimates", sentiment="positive", confidence=0.8, brief_reason="Stronger revenue supports earnings.")
    [item], _, llm = score([BEATS], [None], [answer])

    assert item.scored_by == "llm" and item.probabilities is None
    assert (item.sentiment, item.confidence, item.score) == ("positive", 0.8, 0.8)
    # The model shortened the title; the record keeps the original.
    assert item.headline == BEATS.title
    assert llm.requests[0]["prompt"] == "headline_sentiment_llm"
    assert llm.requests[0]["variables"]["publisher"] == "unknown"


def test_nobody_decides_when_both_fail():
    [item], _, _ = score([BEATS], [None], [None])

    assert item.scored_by == "none" and item.score is None
    assert (item.sentiment, item.confidence) == ("neutral", 0.0)


# --- the Sentiment score ----------------------------------------------------


def jev_item(pos: float, neu: float, neg: float) -> HeadlineSentiment:
    probabilities = {"positive": pos, "neutral": neu, "negative": neg}
    label = max(probabilities, key=probabilities.get)
    return HeadlineSentiment(
        headline="h", sentiment=label, confidence=0.5, brief_reason="r",
        probabilities=probabilities, scored_by="jev", score=round(pos - neg, 4),
    )


def test_score_is_the_mean_of_headline_scores():
    # +0.80, -0.63 and +0.02: the mean is +0.0633, inside the neutral band.
    items = [jev_item(0.85, 0.10, 0.05), jev_item(0.05, 0.27, 0.68), jev_item(0.06, 0.90, 0.04)]

    warnings: list[str] = []

    summary = summarise_sentiment(items, warnings)

    assert summary.score == pytest.approx(0.0633, abs=1e-4)
    assert summary.label == "neutral"
    assert (summary.scored, summary.by_jev, summary.by_llm, summary.failed) == (3, 3, 0, 0)
    assert summary.counts == {"positive": 1, "negative": 1, "neutral": 1}
    assert warnings == []


def test_llm_scores_count_and_failures_are_left_out():
    llm_item = HeadlineSentiment(headline="h", sentiment="negative", confidence=0.7, brief_reason="r", probabilities=None, scored_by="llm", score=-0.7)
    failed = HeadlineSentiment(headline="h", sentiment="neutral", confidence=0.0, brief_reason="r", probabilities=None, scored_by="none", score=None)

    summary = summarise_sentiment([jev_item(0.85, 0.10, 0.05), llm_item, failed], [])

    assert summary.score == pytest.approx((0.80 - 0.70) / 2)
    assert (summary.scored, summary.by_jev, summary.by_llm, summary.failed) == (2, 1, 1, 1)
    assert summary.counts == {"positive": 1, "negative": 1, "neutral": 0}


@pytest.mark.parametrize(
    ("pos", "neg", "label"),
    [(0.60, 0.39, "positive"), (0.40, 0.20, "neutral"), (0.20, 0.40, "neutral"), (0.39, 0.60, "negative")],
)
def test_label_uses_the_threshold(pos, neg, label):
    # Scores +0.21, +0.20, -0.20 and -0.21: exactly on the threshold is still neutral.
    summary = summarise_sentiment([jev_item(pos, 1 - pos - neg, neg)], [])

    assert summary.label == label
    assert SENTIMENT_LABEL_THRESHOLD == 0.2


def test_no_scored_headlines_gives_no_score_and_a_warning():
    warnings: list[str] = []

    summary = summarise_sentiment([], warnings)

    assert summary.score is None and summary.label is None
    assert any("no headline" in w for w in warnings)
