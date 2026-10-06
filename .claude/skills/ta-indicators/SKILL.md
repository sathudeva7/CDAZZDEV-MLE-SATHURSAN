---
name: ta-indicators
description: Use, add or change technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, volatility) or the momentum signal in task1_financial/. Use when touching indicators.py or signals.py, or when reading indicator values for the Task 1A summary, the Task 1B prompt, or Task 3's get_price_data and calculate_volatility tools.
---

# Technical indicators

Indicator accuracy is the largest single criterion in Task 1A, and the
interviewer will ask why each number is right. The conventions (seeding, standard deviation, edge
values) live in the docstring of `task1_financial/indicators.py`, and the
voting rules in the docstring of `task1_financial/signals.py`. Read the
relevant one before changing either file, and keep it true in the same change.

## Using them

1. **Clean prices first.** `add_indicators` expects:
   - prices in ascending date order, one row per trading day
   - NaN rows dropped, with the count logged
   - adjusted prices: yfinance's default `auto_adjust=True`, with
     `multi_level_index=False` to get flat columns

   Two years of data (about 500 rows) covers SMA-200.
   `task1_financial/data.py`'s `fetch_ohlcv` returns prices in this shape, so
   use it (or `pipeline.run_market_data`, which runs every step below) rather
   than calling yfinance directly.
2. `df = add_indicators(ohlcv)`, then `signal = momentum_signal(df)`.
3. Read columns through the `COL_*` constants. Task 3 imports these
   functions, so every indicator value in the repo comes from this one module.
4. `signal.model_dump()` goes into the summary dict and the Task 1B prompt as-is.

## Adding or changing an indicator

1. Add a named constant for every window, period and threshold, and a `COL_*`
   name built from it.
2. Write it with pandas and numpy only. Recursive smoothing (EMA or Wilder)
   goes through `_seeded_smoothing`. The pandas defaults that break the
   conventions are listed in the module docstring.
3. Test it three ways in `tests/test_indicators.py`:
   - **Published values.** When a reference table exists, such as a
     StockCharts ChartSchool spreadsheet, extract it into `tests/fixtures/`
     with a `# SOURCE:` header line and a `CITATIONS.md` row.
   - **A hand-checked example**, small enough to work through on paper.
   - **A property**: bounds, a flat price, or a short series giving NaN
     without raising.
4. **Mutation check.** Plant the convention's classic mistake, confirm a test
   goes red, then restore the code. Examples: `ddof=1`, the pandas
   first-price EMA seed, a simple average in place of Wilder smoothing.
5. Update the convention list in the module docstring.

## Changing the momentum signal

1. Edit the constants or votes in `signals.py`, and update its docstring.
2. Add a row to `test_score_maps_to_label` for each label the change can produce.
3. When the `MomentumSignal` shape changes, bump the `version` of every Task 1B
   prompt that reads it.

Done when `pytest -q` passes, the docstrings match the code, and the mutation
check has been run for any changed formula.
