"""Pydantic schemas for Task 1B: what the LLM must answer, and the records built from it.

Schemas the LLM answers (sent as a strict JSON schema by common/llm.py):
HeadlineReason, HeadlineSentimentAnswer and Recommendation. Every field is
required and described, because the description reaches the model.

Records this package builds (never sent to a model): HeadlineSentiment and
SentimentSummary. See GLOSSARY.md for the terms.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

Sentiment = Literal["positive", "negative", "neutral"]
# Who decided a headline's sentiment: Jev, the LLM when Jev failed, or nobody.
ScoredBy = Literal["jev", "llm", "none"]
RecommendationLabel = Literal["Buy", "Hold", "Sell"]

MIN_JUSTIFICATION_SENTENCES = 3
MAX_JUSTIFICATION_SENTENCES = 5
MIN_KEY_FACTORS = 2
MAX_KEY_FACTORS = 4

# A sentence ends at . ! or ? followed by whitespace and a capital letter. A decimal
# point ("4.2%") has no space after it, and "vs. its" is followed by a lower-case letter.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def count_sentences(text: str) -> int:
    return len([part for part in SENTENCE_BREAK.split(text.strip()) if part])


# --- answers the LLM gives -------------------------------------------------


class HeadlineReason(BaseModel):
    """The LLM's explanation of a sentiment label Jev already chose."""

    brief_reason: str = Field(
        description="One sentence of at most 30 words on why the headline is likely to move the share price this way"
    )


class HeadlineSentimentAnswer(BaseModel):
    """The LLM's full judgement of one headline, used only when Jev is unavailable."""

    headline: str = Field(description="The headline exactly as given")
    sentiment: Sentiment = Field(description="Likely effect on the company's share price")
    confidence: float = Field(ge=0, le=1, description="How sure you are of the sentiment, from 0 to 1")
    brief_reason: str = Field(description="One sentence of at most 30 words explaining the sentiment")


class Recommendation(BaseModel):
    """The LLM's Buy, Hold or Sell call, reasoned over indicator combinations."""

    recommendation: RecommendationLabel = Field(description="Buy, Hold or Sell over the stated horizon")
    justification: str = Field(
        description="Three to five sentences; each one connects at least two of the facts given, and none uses abbreviations"
    )
    key_factors: list[str] = Field(
        min_length=MIN_KEY_FACTORS,
        max_length=MAX_KEY_FACTORS,
        description="Two to four short phrases, each naming the indicators it combines and what the combination means",
    )

    @field_validator("justification")
    @classmethod
    def three_to_five_sentences(cls, value: str) -> str:
        sentences = count_sentences(value)
        if not MIN_JUSTIFICATION_SENTENCES <= sentences <= MAX_JUSTIFICATION_SENTENCES:
            raise ValueError(
                f"justification has {sentences} sentences; write between "
                f"{MIN_JUSTIFICATION_SENTENCES} and {MAX_JUSTIFICATION_SENTENCES}"
            )
        return value


# --- records built from the answers ----------------------------------------


class HeadlineSentiment(BaseModel):
    """One headline's sentiment: the brief's four fields, plus where the decision came from."""

    headline: str
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1)
    brief_reason: str
    probabilities: dict[str, float] | None = Field(description="Jev's probability per label; None when the LLM decided")
    scored_by: ScoredBy
    score: float | None = Field(ge=-1, le=1, description="p(positive) - p(negative); None when nobody could score it")


class SentimentSummary(BaseModel):
    """The Sentiment score: the mean headline score, from -1 to +1, and how it was reached."""

    score: float | None = Field(description="Mean headline score; None when no headline was scored")
    label: Sentiment | None
    scored: int = Field(description="Headlines that count toward the score")
    by_jev: int
    by_llm: int
    failed: int = Field(description="Headlines nobody could score, left out of the score")
    counts: dict[str, int] = Field(description="Scored headlines per sentiment label")
