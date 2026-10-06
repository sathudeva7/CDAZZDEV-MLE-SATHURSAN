"""Technical indicators computed from first principles with pandas and numpy.

No TA-Lib or indicator library is used. Each convention below matches the
StockCharts ChartSchool worked examples in tests/fixtures/, and the tests check
the values to 1e-6.

- SMA: plain rolling mean, NaN until `window` prices exist.
- EMA: multiplier k = 2 / (span + 1), seeded with the SMA of the first `span`
  values (StockCharts). pandas `ewm(adjust=False)` on its own seeds with the
  first price instead, which disagrees for the first 60-90 bars of a 26-day EMA.
- RSI: Wilder smoothing. The first average gain and loss are the plain mean of
  the first `period` price changes; after that, average = (previous average *
  (period - 1) + today) / period. RSI = 100 - 100 / (1 + avg gain / avg loss).
  The first value lands on bar period + 1. With no losses RSI is 100; with
  neither gains nor losses (a flat price) it is 50.
- MACD: EMA(fast) - EMA(slow). The signal line is the EMA(signal) of the MACD
  line, seeded the same way, from the first `signal` MACD values. StockCharts
  documents no seed for the signal line, so the EMA rule above is applied for
  consistency: MACD starts on bar `slow`, the signal on bar slow + signal - 1.
  Histogram = MACD - signal.
- Bollinger Bands: middle = SMA(window), bands = middle +/- num_std * standard
  deviation. The standard deviation is the population one (ddof=0), as John
  Bollinger and StockCharts specify; pandas defaults to the sample one (ddof=1),
  which is wrong here. %b = (close - lower) / (upper - lower), and 0.5 when the
  band has zero width (a flat price).
- Volatility: sample standard deviation (ddof=1) of daily log returns over
  `window` days, annualised by sqrt(252 trading days).

Inputs are prices in ascending date order, one row per trading day, with gaps
already removed by the fetching code. Data problems never raise: a series too
short for an indicator gives NaN and a logged warning. Caller mistakes raise:
TypeError for something that is not a DataFrame, ValueError for a missing
price column.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the ta-indicators skill with its indicator and signal modules and tests', Date: 2026-10-06

from __future__ import annotations

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SMA_SHORT_WINDOW = 50
SMA_LONG_WINDOW = 200
RSI_PERIOD = 14
MACD_FAST_SPAN = 12
MACD_SLOW_SPAN = 26
MACD_SIGNAL_SPAN = 9
BB_WINDOW = 20
BB_NUM_STD = 2.0
VOLATILITY_WINDOW = 30
TRADING_DAYS_PER_YEAR = 252
# Population standard deviation, per John Bollinger. pandas' default is 1 (sample).
BB_DDOF = 0
# Sample standard deviation, the convention for historical volatility.
VOLATILITY_DDOF = 1
# %b when the band has zero width: the price sits on the middle band.
FLAT_BAND_PCT_B = 0.5
# RSI when the price neither rose nor fell over the whole smoothing window.
FLAT_PRICE_RSI = 50.0
MAX_RSI = 100.0
# Band widths below this count as zero, absorbing floating-point noise.
ZERO_WIDTH_TOLERANCE = 1e-12

DEFAULT_PRICE_COLUMN = "Close"

# Column names written by add_indicators, built from the constants above.
COL_SMA_SHORT = f"sma_{SMA_SHORT_WINDOW}"
COL_SMA_LONG = f"sma_{SMA_LONG_WINDOW}"
COL_RSI = f"rsi_{RSI_PERIOD}"
COL_MACD = "macd"
COL_MACD_SIGNAL = "macd_signal"
COL_MACD_HIST = "macd_hist"
COL_BB_MIDDLE = "bb_middle"
COL_BB_UPPER = "bb_upper"
COL_BB_LOWER = "bb_lower"
COL_BB_PCT_B = "bb_pct_b"
COL_VOLATILITY = f"hv_{VOLATILITY_WINDOW}"


def sma(close: pd.Series, window: int) -> pd.Series:
    """Simple moving average of the last `window` prices."""
    _warn_if_short(close, window, f"SMA({window})")
    return close.rolling(window, min_periods=window).mean()


def ema(values: pd.Series, span: int) -> pd.Series:
    """Exponential moving average, seeded with the SMA of the first `span` values.

    Leading NaNs are skipped, so this also smooths a series that starts late,
    such as the MACD line when building the signal line.
    """
    _warn_if_short(values, span, f"EMA({span})")
    return _ema(values, span)


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Relative Strength Index with Wilder smoothing, from 0 to 100."""
    _warn_if_short(close, period + 1, f"RSI({period})")
    change = close.diff()
    gains = change.clip(lower=0)
    losses = (-change).clip(lower=0)
    # Wilder smoothing is an EMA with alpha = 1 / period, seeded with a plain mean.
    avg_gain = _seeded_smoothing(gains, window=period, alpha=1 / period)
    avg_loss = _seeded_smoothing(losses, window=period, alpha=1 / period)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = 100 - 100 / (1 + avg_gain / avg_loss)
    no_losses = avg_loss == 0
    flat = no_losses & (avg_gain == 0)
    result = raw.mask(no_losses, MAX_RSI).mask(flat, FLAT_PRICE_RSI)
    return result.where(avg_gain.notna())


def macd(
    close: pd.Series,
    fast: int = MACD_FAST_SPAN,
    slow: int = MACD_SLOW_SPAN,
    signal: int = MACD_SIGNAL_SPAN,
) -> pd.DataFrame:
    """MACD line, signal line and histogram, as columns macd, signal and hist."""
    _warn_if_short(close, slow + signal - 1, f"MACD({fast},{slow},{signal}) signal line")
    line = _ema(close, fast) - _ema(close, slow)
    signal_line = _ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": signal_line, "hist": line - signal_line})


def bollinger(close: pd.Series, window: int = BB_WINDOW, num_std: float = BB_NUM_STD) -> pd.DataFrame:
    """Bollinger Bands as columns middle, upper, lower and pct_b."""
    middle = sma(close, window)
    std = close.rolling(window, min_periods=window).std(ddof=BB_DDOF)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = (close - lower) / width
    pct_b = pct_b.mask(width.abs() < ZERO_WIDTH_TOLERANCE, FLAT_BAND_PCT_B).where(width.notna())
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower, "pct_b": pct_b})


def annualised_volatility(close: pd.Series, window: int = VOLATILITY_WINDOW) -> pd.Series:
    """Annualised historical volatility from daily log returns, as a fraction (0.25 = 25%)."""
    _warn_if_short(close, window + 1, f"volatility({window})")
    log_returns = np.log(close / close.shift(1))
    daily = log_returns.rolling(window, min_periods=window).std(ddof=VOLATILITY_DDOF)
    return daily * math.sqrt(TRADING_DAYS_PER_YEAR)


def add_indicators(ohlcv: pd.DataFrame, price_column: str = DEFAULT_PRICE_COLUMN) -> pd.DataFrame:
    """A copy of `ohlcv` with every indicator added as a column.

    The columns are named by the COL_* constants in this module.
    """
    if not isinstance(ohlcv, pd.DataFrame):
        raise TypeError(f"expected a DataFrame, got {type(ohlcv).__name__}")
    if price_column not in ohlcv.columns:
        raise ValueError(f"column {price_column!r} not found; columns are {list(ohlcv.columns)}")

    close = ohlcv[price_column].astype(float)
    out = ohlcv.copy()
    out[COL_SMA_SHORT] = sma(close, SMA_SHORT_WINDOW)
    out[COL_SMA_LONG] = sma(close, SMA_LONG_WINDOW)
    out[COL_RSI] = rsi(close, RSI_PERIOD)
    macd_frame = macd(close, MACD_FAST_SPAN, MACD_SLOW_SPAN, MACD_SIGNAL_SPAN)
    out[COL_MACD] = macd_frame["macd"]
    out[COL_MACD_SIGNAL] = macd_frame["signal"]
    out[COL_MACD_HIST] = macd_frame["hist"]
    bands = bollinger(close, BB_WINDOW, BB_NUM_STD)
    out[COL_BB_MIDDLE] = bands["middle"]
    out[COL_BB_UPPER] = bands["upper"]
    out[COL_BB_LOWER] = bands["lower"]
    out[COL_BB_PCT_B] = bands["pct_b"]
    out[COL_VOLATILITY] = annualised_volatility(close, VOLATILITY_WINDOW)
    return out


def _ema(values: pd.Series, span: int) -> pd.Series:
    return _seeded_smoothing(values, window=span, alpha=2 / (span + 1))


def _seeded_smoothing(values: pd.Series, window: int, alpha: float) -> pd.Series:
    """Recursive smoothing y_t = y_(t-1) + alpha * (x_t - y_(t-1)), seeded with a plain mean.

    The seed is the mean of the first `window` non-NaN values and lands on the
    last of them. From there pandas `ewm(adjust=False)` runs the recursion,
    because it starts from whatever its first input is: here, the seed.
    """
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna()
    if len(valid) < window:
        return result
    seeded = valid.iloc[window - 1 :].astype(float).copy()
    seeded.iloc[0] = valid.iloc[:window].mean()
    result.loc[seeded.index] = seeded.ewm(alpha=alpha, adjust=False).mean()
    return result


def _warn_if_short(close: pd.Series, needed: int, name: str) -> None:
    if close.notna().sum() < needed:
        logger.warning(
            "%d prices is too few for %s (needs %d); returning NaN", close.notna().sum(), name, needed
        )
