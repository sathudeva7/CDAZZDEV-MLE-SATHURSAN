<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Ship the executed Task 1 notebook with its README, call log and report', Date: 2026-10-07 -->
# Task 1: LLM-powered equity research assistant

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sathudeva7/CDAZZDEV-MLE-SATHURSAN/blob/main/task1_financial/task1_equity_research.ipynb)

The pipeline does four things for a ticker (AAPL in the submitted run):
- fetches a year of prices from yfinance
- computes SMA, EMA, RSI, MACD and Bollinger Bands from first principles in pandas
- collects recent headlines from Yahoo and Google News RSS
- asks an LLM for structured, validated JSON: one sentiment label and reason per headline,
  and a Buy/Hold/Sell Recommendation justified from the indicators

**Bonus:** a rendered brief.
[View the HTML report](https://htmlpreview.github.io/?https://github.com/sathudeva7/CDAZZDEV-MLE-SATHURSAN/blob/main/task1_financial/reports/AAPL_brief.html),
or read the [Markdown version](reports/AAPL_brief.md).

## Submitted run (Colab, `free` profile, 39 s)

| Output | Value |
|---|---|
| Recommendation | Hold |
| Sentiment score | -0.41 (negative) |
| Headlines | 15 retrieved; 14 labelled by the LLM, 1 failed and was left out of the score |
| Logged requests | 31: 15 ok, 16 fallbacks |

**Why 16 fallbacks:**
- **15 of them are Jev.** Jev's key was not set, so each Jev request fell back on purpose and
  the Groq LLM labelled the headlines instead (`docs/adr/0001`).
- **1 is a headline** where the LLM's answers failed validation twice. It is labelled as a
  fallback and kept out of the Sentiment score.

Every request is in [`logs/llm_calls.jsonl`](logs/llm_calls.jsonl).

## Where the logic lives

| File | What it does |
|---|---|
| `data.py`, `summary.py` | yfinance fetch with retries, and the summary dictionary |
| `indicators.py`, `signals.py` | the indicators and the momentum signal |
| `news.py` | headline retrieval and de-duplication |
| `sentiment.py`, `recommendation.py`, `prompts.py`, `schemas.py` | the LLM calls, through `common/llm.py` |
| `report.py` | the bonus Markdown and HTML brief, with a chart |
| `pipeline.py` | `run_market_data` (1A) and `run_analysis` (1B) |

## Running

- **Notebook:** open it with the badge, add `GROQ_API_KEY` under Colab's Secrets, and
  optionally `OPENROUTER_API_KEY` and `JEV_API_KEY`. Then Runtime → Run all.
- **Tests:** `pytest` runs the offline tests; `pytest -m live` also calls the real APIs.
