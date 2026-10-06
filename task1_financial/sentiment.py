"""Headline sentiment and the Sentiment score (see GLOSSARY.md and docs/adr/0001).

For each headline:

1. Jev decides the label from the company, publisher and title, with a
   probability for every label.
2. The LLM writes brief_reason for Jev's label. If that call fails, the
   decision stands with a placeholder reason, because the decision is Jev's.
3. If Jev fails, the LLM makes the whole judgement (scored_by="llm"). If that
   fails too, the headline is neutral with confidence 0 and left out of the
   score (scored_by="none").

Headline score: p(positive) - p(negative) from Jev's probabilities, so a
71% negative / 29% neutral answer scores -0.71. The LLM gives no
probabilities, so its score is the label's sign times its confidence
(negative at 0.7 scores -0.7). This confidence is the model's own estimate,
not a measured probability, which is why Jev decides whenever it can.

Sentiment score: the mean of the headline scores, from -1 to +1. Neutral
headlines count, so mostly unremarkable news pulls the score toward 0.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import get_args

from common.jev import JevClient
from common.llm import StructuredLLM
from task1_financial.news import Headline
from task1_financial.prompts import (
    HEADLINE_REASON,
    HEADLINE_SENTIMENT,
    HEADLINE_SENTIMENT_QUESTION,
)
from task1_financial.schemas import (
    HeadlineReason,
    HeadlineSentiment,
    HeadlineSentimentAnswer,
    Sentiment,
    SentimentSummary,
)

logger = logging.getLogger(__name__)

# A Sentiment score above +0.2 is positive and below -0.2 negative; in between is neutral.
SENTIMENT_LABEL_THRESHOLD = 0.2
SCORE_DECIMALS = 4
SENTIMENT_LABELS: tuple[str, ...] = get_args(Sentiment)
SIGN = {"positive": 1, "neutral": 0, "negative": -1}
UNKNOWN_PUBLISHER = "unknown"
REASON_UNAVAILABLE = "No reason available: the LLM call failed."
NOT_SCORED_REASON = "Not scored: Jev and the LLM both failed."

# The reason restates a decision, so low effort is enough; reasoning tokens count toward rate limits.
REASON_EFFORT = "low"
FALLBACK_SENTIMENT_EFFORT = "low"


def score_headlines(
    headlines: Sequence[Headline], ticker: str, company: str | None, jev: JevClient, llm: StructuredLLM
) -> list[HeadlineSentiment]:
    """Headline sentiment for each headline, in the same order. Never raises for a model failure."""
    return [_score_headline(headline, ticker, company or ticker, jev, llm) for headline in headlines]


def summarise_sentiment(items: Sequence[HeadlineSentiment], warnings: list[str]) -> SentimentSummary:
    """The Sentiment score over every headline that was scored; failed headlines are only counted."""
    scored = [item for item in items if item.score is not None]
    counts = {label: sum(item.sentiment == label for item in scored) for label in SENTIMENT_LABELS}
    score = round(sum(item.score for item in scored) / len(scored), SCORE_DECIMALS) if scored else None
    if score is None:
        _warn(warnings, "no headline could be scored, so there is no Sentiment score")
    failed = len(items) - len(scored)
    if failed and scored:
        logger.info("%d of %d headlines could not be scored and are left out of the Sentiment score", failed, len(items))
    return SentimentSummary(
        score=score,
        label=_label(score),
        scored=len(scored),
        by_jev=sum(item.scored_by == "jev" for item in scored),
        by_llm=sum(item.scored_by == "llm" for item in scored),
        failed=failed,
        counts=counts,
    )


def _score_headline(headline: Headline, ticker: str, company: str, jev: JevClient, llm: StructuredLLM) -> HeadlineSentiment:
    variables = {
        "company": company,
        "ticker": ticker,
        "publisher": headline.publisher or UNKNOWN_PUBLISHER,
        "headline": headline.title,
    }
    decision = jev.choose(HEADLINE_SENTIMENT_QUESTION, variables).value
    if decision is not None:
        reason = llm.call(
            HEADLINE_REASON,
            {**variables, "sentiment": decision.choice, "probabilities": _describe(decision.probabilities)},
            HeadlineReason,
            fallback=HeadlineReason(brief_reason=REASON_UNAVAILABLE),
            reasoning_effort=REASON_EFFORT,
        )
        probabilities = {label: decision.probabilities[label] for label in SENTIMENT_LABELS}
        return HeadlineSentiment(
            headline=headline.title,
            sentiment=decision.choice,
            confidence=decision.confidence,
            brief_reason=reason.value.brief_reason,
            probabilities=probabilities,
            scored_by="jev",
            score=round(probabilities["positive"] - probabilities["negative"], SCORE_DECIMALS),
        )

    answer = llm.call(
        HEADLINE_SENTIMENT,
        variables,
        HeadlineSentimentAnswer,
        fallback=HeadlineSentimentAnswer(headline=headline.title, sentiment="neutral", confidence=0.0, brief_reason=NOT_SCORED_REASON),
        reasoning_effort=FALLBACK_SENTIMENT_EFFORT,
    )
    if answer.ok and answer.value.headline != headline.title:
        # The record must match its source, so the original title is kept.
        logger.info("LLM changed the headline text; keeping the original: %r", headline.title)
    return HeadlineSentiment(
        headline=headline.title,
        sentiment=answer.value.sentiment,
        confidence=answer.value.confidence,
        brief_reason=answer.value.brief_reason,
        probabilities=None,
        scored_by="llm" if answer.ok else "none",
        score=SIGN[answer.value.sentiment] * answer.value.confidence if answer.ok else None,
    )


def _describe(probabilities: dict[str, float]) -> str:
    """'negative 71%, neutral 29%, positive 0%': most likely first, for the reason prompt."""
    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    return ", ".join(f"{label} {probability:.0%}" for label, probability in ordered)


def _label(score: float | None) -> Sentiment | None:
    if score is None:
        return None
    if score > SENTIMENT_LABEL_THRESHOLD:
        return "positive"
    if score < -SENTIMENT_LABEL_THRESHOLD:
        return "negative"
    return "neutral"


def _warn(warnings: list[str], message: str) -> None:
    """Log a fallback and record it for the analysis, so the two always agree."""
    logger.warning(message)
    warnings.append(message)
