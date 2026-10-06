"""Tests for task1_financial/indicators.py.

Three kinds of evidence that the numbers are right:
1. Published values: StockCharts ChartSchool worked examples (tests/fixtures/).
2. Hand-checked small examples, small enough to verify on paper.
3. Properties that must hold for any input: bounds, flat prices, short series.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the ta-indicators skill with its indicator and signal modules and tests', Date: 2026-10-06

from __future__ import annotations

import itertools
import json
import logging
import math
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from task1_financial.indicators import (
    COL_RSI,
    COL_SMA_LONG,
    COL_VOLATILITY,
    TRADING_DAYS_PER_YEAR,
    add_indicators,
    annualised_volatility,
    bollinger,
    ema,
    macd,
    rsi,
    sma,
)

FIXTURES = Path(__file__).parent / "fixtures"
TOLERANCE = 1e-6


def stockcharts(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES / name, comment="#", parse_dates=["date"], index_col="date")


def assert_matches_reference(actual: pd.Series, expected: pd.Series) -> None:
    """Equal to TOLERANCE where the reference has a value, NaN where it is blank."""
    defined = expected.notna()
    assert defined.any()
    np.testing.assert_allclose(actual[defined].to_numpy(), expected[defined].to_numpy(), atol=TOLERANCE, rtol=0)
    assert actual[~defined].isna().all(), "values appeared during warm-up"


def wave(n: int) -> pd.Series:
    """A deterministic, non-trivial price path: drift plus a cycle."""
    i = np.arange(n)
    return pd.Series(100 + 0.3 * i + 10 * np.sin(i / 5), index=pd.bdate_range("2024-01-01", periods=n))


# 1. Published values -------------------------------------------------------


def test_sma_matches_stockcharts():
    ref = stockcharts("stockcharts_movavg.csv")
    assert_matches_reference(sma(ref["close"], 10), ref["sma_10"])


def test_ema_matches_stockcharts():
    ref = stockcharts("stockcharts_movavg.csv")
    assert_matches_reference(ema(ref["close"], 10), ref["ema_10"])


def test_rsi_matches_stockcharts():
    ref = stockcharts("stockcharts_rsi.csv")
    assert_matches_reference(rsi(ref["close"], 14), ref["rsi_14"])


def test_bollinger_matches_stockcharts():
    ref = stockcharts("stockcharts_bbands.csv")
    bands = bollinger(ref["close"], 20, 2)
    assert_matches_reference(bands["middle"], ref["bb_middle"])
    assert_matches_reference(bands["upper"], ref["bb_upper"])
    assert_matches_reference(bands["lower"], ref["bb_lower"])


# 2. Hand-checked examples --------------------------------------------------


def test_ema_hand_example():
    # span 3 -> k = 0.5, seeded with the SMA of the first three prices (11.0).
    result = ema(pd.Series([10, 12, 11, 13, 15, 14, 20], dtype=float), 3)
    assert result.isna().tolist()[:2] == [True, True]
    assert result.iloc[2:].tolist() == pytest.approx([11.0, 12.0, 13.5, 13.75, 16.875])


def test_rsi_hand_example():
    # Period 3. Changes +1, -0.5, +1 seed the averages at 2/3 and 1/6 (RSI 80);
    # Wilder smoothing then gives 84.6, and one -1 day pulls it back to 50.
    result = rsi(pd.Series([10, 11, 10.5, 11.5, 12, 11]), 3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3:].tolist() == pytest.approx([80.0, 100 - 100 / 6.5, 50.0])


def test_bollinger_uses_population_standard_deviation():
    # Prices 20..24 have mean 22 and squared deviations summing to 10.
    # Population SD = sqrt(10 / 5); the sample SD sqrt(10 / 4) would widen the bands.
    bands = bollinger(pd.Series([20, 22, 21, 23, 24], dtype=float), 5, 2)
    last = bands.iloc[-1]
    assert last["middle"] == pytest.approx(22.0)
    assert last["upper"] == pytest.approx(22 + 2 * math.sqrt(2))
    assert last["lower"] == pytest.approx(22 - 2 * math.sqrt(2))
    assert last["pct_b"] == pytest.approx((24 - (22 - 2 * math.sqrt(2))) / (4 * math.sqrt(2)))


def test_volatility_is_sample_sd_of_log_returns_annualised():
    prices = [100.0, 102.0, 99.0, 101.0]
    log_returns = [math.log(b / a) for a, b in itertools.pairwise(prices)]
    expected = statistics.stdev(log_returns) * math.sqrt(TRADING_DAYS_PER_YEAR)
    result = annualised_volatility(pd.Series(prices), 3)
    assert result.iloc[:3].isna().all()
    assert result.iloc[3] == pytest.approx(expected)


def test_macd_is_built_from_the_tested_ema():
    close = wave(80)
    frame = macd(close, 12, 26, 9)
    line = frame["macd"]

    pd.testing.assert_series_equal(line, ema(close, 12) - ema(close, 26), check_names=False)
    assert line.first_valid_index() == close.index[25]  # bar 26
    assert frame["signal"].first_valid_index() == close.index[33]  # bar 26 + 9 - 1
    # The signal line is seeded with the mean of the first nine MACD values.
    assert frame["signal"].iloc[33] == pytest.approx(line.iloc[25:34].mean())
    pd.testing.assert_series_equal(frame["hist"], line - frame["signal"], check_names=False)


# 3. Properties ---------------------------------------------------------------


def test_rsi_stays_between_0_and_100_on_a_random_walk():
    rng = np.random.default_rng(seed=7)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))))
    values = rsi(close, 14).dropna()
    assert len(values) == 500 - 14
    assert values.between(0, 100).all()


def test_flat_price_gives_the_documented_edge_values():
    close = pd.Series([50.0] * 60)
    bands = bollinger(close, 20, 2).iloc[-1]
    assert sma(close, 20).iloc[-1] == 50.0
    assert rsi(close, 14).iloc[-1] == 50.0
    assert bands["upper"] == bands["lower"] == bands["middle"] == 50.0
    assert bands["pct_b"] == 0.5
    assert annualised_volatility(close, 30).iloc[-1] == 0.0


def test_rsi_is_100_when_there_are_no_losses():
    assert rsi(pd.Series(np.arange(1.0, 31.0)), 14).dropna().eq(100.0).all()


@pytest.mark.parametrize("length", [0, 1, 10])
def test_short_series_give_nan_and_a_warning_without_raising(length, caplog):
    close = pd.Series(np.linspace(10, 20, length))
    with caplog.at_level(logging.WARNING, logger="task1_financial.indicators"):
        out = add_indicators(pd.DataFrame({"Close": close}))
    assert out[COL_SMA_LONG].isna().all() and out[COL_RSI].isna().all()
    assert "too few" in caplog.text


def test_add_indicators_adds_every_column_and_leaves_input_untouched():
    ohlcv = pd.DataFrame({"Close": wave(260), "Volume": 1_000})
    out = add_indicators(ohlcv)
    expected = {
        "sma_50", "sma_200", "rsi_14", "macd", "macd_signal", "macd_hist",
        "bb_middle", "bb_upper", "bb_lower", "bb_pct_b", "hv_30",
    }
    assert expected <= set(out.columns)
    assert list(ohlcv.columns) == ["Close", "Volume"]
    assert out.iloc[-1][sorted(expected)].notna().all()
    assert out[COL_VOLATILITY].iloc[-1] > 0
    json.dumps(out.iloc[-1][sorted(expected)].to_dict())  # plain floats, ready for a summary dict


def test_add_indicators_rejects_a_frame_without_the_price_column():
    with pytest.raises(ValueError, match="'Close' not found"):
        add_indicators(pd.DataFrame({"Open": [1.0, 2.0]}))
