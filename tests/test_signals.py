"""Scenario tests for task1_financial/signals.py: every label, flag and missing-data path."""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the ta-indicators skill with its indicator and signal modules and tests', Date: 2026-10-06

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from task1_financial.indicators import (
    COL_BB_PCT_B,
    COL_MACD,
    COL_MACD_HIST,
    COL_MACD_SIGNAL,
    COL_RSI,
    COL_SMA_LONG,
    COL_SMA_SHORT,
    add_indicators,
)
from task1_financial.signals import MomentumSignal, momentum_signal

INDEX = pd.to_datetime(["2026-10-01", "2026-10-02"])


def frame(
    close=110.0,
    sma_50=100.0,
    sma_200=90.0,
    macd=2.0,
    macd_signal=1.0,
    hist_prev=0.5,
    rsi=60.0,
    pct_b=0.7,
):
    """Two bars of add_indicators() output. The defaults are bullish on every vote."""
    hist = None if macd is None or macd_signal is None else macd - macd_signal
    last = {
        "Close": close, COL_SMA_SHORT: sma_50, COL_SMA_LONG: sma_200, COL_MACD: macd,
        COL_MACD_SIGNAL: macd_signal, COL_MACD_HIST: hist, COL_RSI: rsi, COL_BB_PCT_B: pct_b,
    }
    prev = {**last, COL_MACD_HIST: hist_prev}
    return pd.DataFrame([prev, last], index=INDEX).astype(float)


def votes(signal: MomentumSignal) -> dict[str, int]:
    return {v.name: v.vote for v in signal.votes}


@pytest.mark.parametrize(
    ("overrides", "label", "score"),
    [
        ({}, "Strong Bullish", 5),
        ({"rsi": 45.0}, "Bullish", 3),
        ({"rsi": 45.0, "close": 95.0}, "Neutral", 1),
        # MACD below signal (-1), but the histogram -0.5 is up from -1.0 (+1).
        ({"rsi": 45.0, "close": 95.0, "macd": 0.5, "hist_prev": -1.0}, "Neutral", -1),
        # Every vote bearish except acceleration: histogram -1.0 is up from -2.0.
        ({"close": 80.0, "sma_200": 110.0, "macd": 0.0, "hist_prev": -2.0, "rsi": 40.0}, "Bearish", -3),
        (
            {"close": 80.0, "sma_200": 110.0, "macd": -2.0, "macd_signal": -1.0, "hist_prev": -0.5, "rsi": 40.0},
            "Strong Bearish",
            -5,
        ),
    ],
)
def test_score_maps_to_label(overrides, label, score):
    signal = momentum_signal(frame(**overrides))
    assert (signal.label, signal.score) == (label, score)
    assert signal.as_of == "2026-10-02"


def test_overbought_is_a_flag_and_leaves_the_score_alone():
    signal = momentum_signal(frame(rsi=75.0, pct_b=1.2))
    assert signal.score == 5 and signal.label == "Strong Bullish"
    assert signal.flags == ["overbought", "above_upper_band"]


def test_oversold_and_below_lower_band_flags():
    signal = momentum_signal(frame(rsi=25.0, pct_b=-0.1))
    assert signal.flags == ["oversold", "below_lower_band"]


def test_missing_long_average_is_unavailable_and_votes_zero():
    signal = momentum_signal(frame(sma_200=np.nan))
    assert votes(signal)["long_term_regime"] == 0
    assert signal.unavailable == ["long_term_regime"]
    assert (signal.score, signal.label) == (4, "Bullish")


def test_single_bar_has_no_macd_acceleration():
    signal = momentum_signal(frame().iloc[[-1]])
    assert signal.unavailable == ["macd_acceleration"]


def test_empty_frame_is_neutral_with_every_vote_unavailable():
    signal = momentum_signal(frame().iloc[0:0])
    assert (signal.label, signal.score, signal.as_of) == ("Neutral", 0, None)
    assert len(signal.unavailable) == 5


def test_equal_values_vote_zero():
    signal = momentum_signal(frame(close=100.0))
    assert votes(signal)["short_term_trend"] == 0
    assert "equals" in signal.votes[0].reason


def test_missing_indicator_columns_are_a_caller_bug():
    with pytest.raises(ValueError, match="run add_indicators"):
        momentum_signal(pd.DataFrame({"Close": [1.0]}))


def test_reasons_cite_the_values_compared():
    reason = momentum_signal(frame()).votes[0].reason
    assert reason == "close 110 is above sma_50 100"


def test_signal_from_real_indicator_output_serialises_to_json():
    # A steady climb: price above both averages, RSI pinned at 100 (overbought).
    close = pd.Series(100 * 1.002 ** np.arange(260), index=pd.bdate_range("2025-01-01", periods=260))
    signal = momentum_signal(add_indicators(pd.DataFrame({"Close": close})))
    assert votes(signal)["short_term_trend"] == 1 and votes(signal)["long_term_regime"] == 1
    assert "overbought" in signal.flags and signal.unavailable == []
    json.dumps(signal.model_dump())
