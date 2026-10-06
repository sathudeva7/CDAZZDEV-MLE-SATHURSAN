"""Momentum signal: a rule-based reading of the latest indicator values.

Direction and stretch are kept apart, so the Task 1B LLM can reason over the
combination (for example "trend up, but overbought") rather than one number.

- Five direction votes, each +1, -1, or 0 (equal values, or the indicator has
  no value yet, in which case it is listed as unavailable):
  short-term trend (close vs SMA-50), long-term regime (SMA-50 vs SMA-200),
  MACD crossover (MACD vs signal), MACD acceleration (histogram vs the day
  before), and RSI momentum (RSI vs 50).
- The score is the sum. +5 is Strong Bullish and +3 Bullish; -3 Bearish and
  -5 Strong Bearish; anything between is Neutral (a mixed 3-2 split, or votes
  missing).
- Stretch flags never change the score: RSI above 70 or below 30, and the close
  above the upper or below the lower Bollinger Band.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the ta-indicators skill with its indicator and signal modules and tests', Date: 2026-10-06

from __future__ import annotations

import math
from typing import Literal

import pandas as pd
from pydantic import BaseModel, Field

from task1_financial.indicators import (
    COL_BB_PCT_B,
    COL_MACD,
    COL_MACD_HIST,
    COL_MACD_SIGNAL,
    COL_RSI,
    COL_SMA_LONG,
    COL_SMA_SHORT,
    DEFAULT_PRICE_COLUMN,
)

RSI_MIDLINE = 50.0
RSI_OVERBOUGHT = 70.0
RSI_OVERSOLD = 30.0
# %b above 1 means the close is above the upper band; below 0, below the lower band.
PCT_B_UPPER = 1.0
PCT_B_LOWER = 0.0
STRONG_SCORE = 5
DIRECTIONAL_SCORE = 3

Label = Literal["Strong Bullish", "Bullish", "Neutral", "Bearish", "Strong Bearish"]
Flag = Literal["overbought", "oversold", "above_upper_band", "below_lower_band"]


class Vote(BaseModel):
    """One direction vote and the values behind it."""

    name: str = Field(description="Which rule cast this vote")
    inputs: dict[str, float | None] = Field(description="The indicator values the rule compared")
    vote: Literal[-1, 0, 1] = Field(description="+1 bullish, -1 bearish, 0 equal or unavailable")
    reason: str = Field(description="One line a reader can check against the inputs")


class MomentumSignal(BaseModel):
    """The momentum reading for the latest bar."""

    as_of: str | None = Field(description="Date of the bar the signal reads")
    label: Label
    score: int = Field(description="Sum of the votes, from -5 to +5")
    votes: list[Vote]
    flags: list[Flag] = Field(description="Stretch warnings; they do not change the score")
    unavailable: list[str] = Field(description="Votes that had no indicator value yet")


def momentum_signal(df: pd.DataFrame, price_column: str = DEFAULT_PRICE_COLUMN) -> MomentumSignal:
    """Read the momentum signal from the last row of add_indicators() output."""
    required = [price_column, COL_SMA_SHORT, COL_SMA_LONG, COL_MACD, COL_MACD_SIGNAL, COL_MACD_HIST, COL_RSI, COL_BB_PCT_B]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"run add_indicators() first; missing columns {missing}")

    last = df.iloc[-1] if len(df) else pd.Series(dtype=float)
    prev = df.iloc[-2] if len(df) > 1 else pd.Series(dtype=float)

    def value(row: pd.Series, col: str) -> float | None:
        raw = row.get(col)
        return None if raw is None or pd.isna(raw) else float(raw)

    close = value(last, price_column)
    votes = [
        _compare("short_term_trend", "close", close, COL_SMA_SHORT, value(last, COL_SMA_SHORT)),
        _compare("long_term_regime", COL_SMA_SHORT, value(last, COL_SMA_SHORT), COL_SMA_LONG, value(last, COL_SMA_LONG)),
        _compare("macd_crossover", COL_MACD, value(last, COL_MACD), COL_MACD_SIGNAL, value(last, COL_MACD_SIGNAL)),
        _compare("macd_acceleration", "hist_today", value(last, COL_MACD_HIST), "hist_yesterday", value(prev, COL_MACD_HIST)),
        _compare("rsi_momentum", COL_RSI, value(last, COL_RSI), "midline", RSI_MIDLINE),
    ]
    score = sum(v.vote for v in votes)
    rsi_now = value(last, COL_RSI)
    pct_b = value(last, COL_BB_PCT_B)

    flags: list[Flag] = []
    if rsi_now is not None and rsi_now > RSI_OVERBOUGHT:
        flags.append("overbought")
    if rsi_now is not None and rsi_now < RSI_OVERSOLD:
        flags.append("oversold")
    if pct_b is not None and pct_b > PCT_B_UPPER:
        flags.append("above_upper_band")
    if pct_b is not None and pct_b < PCT_B_LOWER:
        flags.append("below_lower_band")

    as_of = df.index[-1] if len(df) else None
    return MomentumSignal(
        as_of=as_of.date().isoformat() if hasattr(as_of, "date") else (None if as_of is None else str(as_of)),
        label=_label(score),
        score=score,
        votes=votes,
        flags=flags,
        unavailable=[v.name for v in votes if v.inputs and None in v.inputs.values()],
    )


def _compare(name: str, left_name: str, left: float | None, right_name: str, right: float | None) -> Vote:
    """+1 when left > right, -1 when left < right, 0 when equal or a value is missing."""
    inputs = {left_name: left, right_name: right}
    if left is None or right is None or math.isnan(left) or math.isnan(right):
        return Vote(name=name, inputs=inputs, vote=0, reason=f"unavailable: {left_name} or {right_name} has no value yet")
    if left > right:
        return Vote(name=name, inputs=inputs, vote=1, reason=f"{left_name} {left:.4g} is above {right_name} {right:.4g}")
    if left < right:
        return Vote(name=name, inputs=inputs, vote=-1, reason=f"{left_name} {left:.4g} is below {right_name} {right:.4g}")
    return Vote(name=name, inputs=inputs, vote=0, reason=f"{left_name} equals {right_name} ({left:.4g})")


def _label(score: int) -> Label:
    if score >= STRONG_SCORE:
        return "Strong Bullish"
    if score >= DIRECTIONAL_SCORE:
        return "Bullish"
    if score <= -STRONG_SCORE:
        return "Strong Bearish"
    if score <= -DIRECTIONAL_SCORE:
        return "Bearish"
    return "Neutral"
