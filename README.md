<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write the root README and a Task 1 REFLECTION as decided in the grilling round', Date: 2026-10-07 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'can u add reflectionmd file now (add Task 2 within 600 words)', Date: 2026-10-07 -->
# CDAZZDEV Senior MLE assessment

A first-pass equity research assistant built on free market data and LLMs
(Task 1), an agentic research system that reuses Task 1's pipeline as its tools
(Task 3), and a small code model fine-tuned with QLoRA to write docstrings (Task 2).

| Task | Status | Notebook | Folder |
|---|---|---|---|
| 1. Financial AI: data pipeline, LLM sentiment and signal reasoning, bonus brief | Done | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sathudeva7/CDAZZDEV-MLE-SATHURSAN/blob/main/task1_financial/task1_equity_research.ipynb) | [`task1_financial/`](task1_financial/) |
| 2. Generative AI: QLoRA fine-tune of Qwen2.5-Coder-3B to write docstrings ([model](https://huggingface.co/sathudeva7/qwen2.5-coder-3b-docstrings)) | Done | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sathudeva7/CDAZZDEV-MLE-SATHURSAN/blob/main/task2_genai/task2_docstring_finetune.ipynb) | [`task2_genai/`](task2_genai/) |
| 3. Agentic AI: tool-using and multi-agent research | In progress | | `task3_agentic/` |

Also in the root: [`REFLECTION.md`](REFLECTION.md) (decisions, improvements and
limitations) and [`CITATIONS.md`](CITATIONS.md) (every AI-assisted file and
every adapted source).

## Quick start

**Colab.** Open the badge above, add the keys under Secrets (the key icon,
with "Notebook access" on), then Runtime → Run all. The first cell clones this
repo and installs `requirements.txt`.

| Secret | Needed for | Required |
|---|---|---|
| `GROQ_API_KEY` | every LLM call (Groq free tier) | yes |
| `OPENROUTER_API_KEY` | fallback provider when Groq fails | no |
| `JEV_API_KEY` | headline sentiment labels (see below) | no |

Task 1A (prices, indicators, news, summary) needs no key at all.

Task 2 needs a T4 GPU runtime instead, and only `HF_TOKEN` (write access) to
publish the merged model; see [`task2_genai/README.md`](task2_genai/README.md).

**Local.**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then fill in the keys above
pytest                        # offline tests; `pytest -m live` also hits the real APIs
jupyter lab task1_financial/task1_equity_research.ipynb
```

## Models and cost

Submitted runs use the `free` profile in
[`common/llm_config.py`](common/llm_config.py): Groq's free tier
(`openai/gpt-oss-120b`), with an OpenRouter free model as fallback. Market data
comes from `yfinance`, and news from the Yahoo Finance and Google News RSS feeds.

One deliberate exception to the free-tier setup: **Jev** (TypeSafe's decision
model, a paid API) labels each headline's sentiment, because it returns
measured probabilities instead of a self-reported confidence. The LLM writes
every piece of text, including the reason for each label and the Buy/Hold/Sell
justification. Without `JEV_API_KEY` the LLM labels every headline itself, so
everything runs on free tiers. The decision is recorded in
[`docs/adr/0001-jev-decides-llm-explains.md`](docs/adr/0001-jev-decides-llm-explains.md).

## Engineering choices

- **One path for every LLM call.** [`common/llm.py`](common/llm.py) sends a
  `Prompt` constant (separate system and user messages) with a strict JSON
  schema, validates the answer with Pydantic, retries once with the validation
  error, falls back to the next provider, and finally returns the caller's
  fallback value. Every request is logged to the task's `logs/llm_calls.jsonl`.
- **Degrade, don't crash.** Missing data, empty feeds and failed model calls
  each produce a logged warning and a placeholder, so a run always finishes and
  says what went wrong (try `run_market_data("NOTATICKERZZ")`).
- **Indicators from first principles.** SMA, RSI (Wilder), MACD and Bollinger
  Bands are written with pandas and numpy in
  [`task1_financial/indicators.py`](task1_financial/indicators.py), and
  checked against StockCharts' published worked examples to 1e-6.
- **Facts, not verdicts, for the LLM.** The Recommendation prompt gets computed
  relationships between the indicators, and not the rule-based momentum label,
  so the model has to combine them itself.
- **Gates before every commit.** Offline tests, `ruff`,
  [`scripts/check_secrets.sh`](scripts/check_secrets.sh) and
  [`scripts/check_citations.sh`](scripts/check_citations.sh).

## Repository map

```text
common/              shared by every task: structured LLM helper, provider profiles, Jev client
task1_financial/     Task 1 notebook and modules: data, indicators, news, summary,
                     sentiment, recommendation, report; logs/ and reports/ from the submitted run
task3_agentic/       Task 3 (in progress)
tests/               offline pytest suite, plus opt-in live tests
docs/adr/            architecture decision records
scripts/             secret scan and citation check
GLOSSARY.md          the domain terms used in code and docs
```
