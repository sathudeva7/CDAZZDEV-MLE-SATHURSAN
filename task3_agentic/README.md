<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Ship the executed Task 3 notebook with its trace, call log, cache and README', Date: 2026-10-07 -->
# Task 3: agentic equity research

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sathudeva7/CDAZZDEV-MLE-SATHURSAN/blob/main/task3_agentic/task3_agentic_research.ipynb)

The query from the brief: analyse AAPL's financial health and market sentiment, name the
top three risks to its share price over the next 90 days, and suggest one data-driven
hedge.

- **3A.** One LangGraph agent picks from five tools: `get_price_data`,
  `calculate_volatility`, `get_news`, `llm_sentiment` and `web_search`. The tools reuse
  Task 1's pipeline.
- **3B.** A data agent hands a typed `DataBrief` to an analyst agent. The analyst can ask
  one typed clarification before it writes the report.
- **3C.** Short-term memory answers follow-up questions, a daily cache avoids repeat
  runs, and every step goes to `agent_trace.jsonl`.

## Submitted run (Colab, `free` profile: Groq `openai/gpt-oss-120b`, 533 s)

| Check | Result |
|---|---|
| 3A report | Written by the LLM, with 3 risks and a `protective_put` hedge |
| Observe and replan | With `get_news` failing, the agent followed the hint to `web_search`, then `llm_sentiment` |
| Bad ticker (`NOTATICKERZZ`) | No exception; the report came from a template, with 4 data gaps listed |
| 3B critique loop | The analyst asked for 6-month price data. The data agent fetched it, and the final report says what it changed |
| 3C short-term memory | The follow-up question was answered with 0 tool calls |
| 3C persistent cache | The second run was a cache hit and returned the same report |
| Trace | 65 events: 30 tool calls, 25 agent turns, 4 reports, 3 cache events, 2 critique events, 1 handoff |
| Tokens | 50,435 in agent turns and 27,058 in structured calls |

The run's files:

| File | What it holds |
|---|---|
| [`logs/agent_trace.jsonl`](logs/agent_trace.jsonl) | Every tool call with its inputs, output, status and duration, plus every agent turn, handoff and critique |
| [`logs/llm_calls.jsonl`](logs/llm_calls.jsonl) | The 9 structured LLM calls |
| [`cache/AAPL_2026-10-07.json`](cache/AAPL_2026-10-07.json) | The cached brief and report |

## Where the logic lives

| File | What it does |
|---|---|
| `tools.py`, `schemas.py` | The five tools, the `{status, data, error, hint}` envelope, and the typed schemas |
| `agent.py` | The hand-written LangGraph loop: budgets (8 tool calls, 6 turns), tool restriction and memory |
| `pipeline.py`, `handoff.py` | The 3B graph: brief, critique, answer and final report |
| `report.py` | The report, hedge levels computed in code, the evidence check and the template fallback |
| `cache.py`, `trace.py`, `printer.py` | The persistent cache, the JSONL trace, and the live printout |
| `dashboard/` | Bonus: a Streamlit viewer for the trace (`streamlit run task3_agentic/dashboard/app.py`) |

## Running

- **Notebook:** open it with the badge, add `GROQ_API_KEY` under Colab's Secrets (and
  optionally `OPENROUTER_API_KEY` for the fallback), then Runtime → Run all.
- **Tests:** `pytest tests/test_task3_*.py` runs offline.
