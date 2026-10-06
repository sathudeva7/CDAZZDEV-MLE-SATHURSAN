# Citations

How AI tools and outside code were used in this repository, following Section 2.2
of the assessment brief. Every AI-assisted block also carries an inline
`AI-ASSISTED` comment, and every adapted block a `SOURCE` comment.

## AI tools used

- **Claude Code (claude-opus-5-5)**: code generation, repository tooling and
  documentation drafts. Each use is listed below.

## AI-assisted code

| Date | Tool (model) | File | Prompt | What it produced |
|---|---|---|---|---|
| 2026-10-06 | Claude (claude-opus-5-5) | `scripts/check_secrets.sh` | 'create the ship skill' | Pre-commit scan for API keys, tokens and .env files |
| 2026-10-06 | Claude (claude-opus-5-5) | `.claude/skills/ship/SKILL.md` | 'create the ship skill' | Claude Code skill: commit gate (tests, secret scan, citations) |
| 2026-10-06 | Claude (claude-opus-5-5) | `scripts/check_citations.sh` | 'Create the cite-ai-usage skill' | Checker for citation marker format and CITATIONS.md coverage |
| 2026-10-06 | Claude (claude-opus-5-5) | `.claude/skills/cite-ai-usage/SKILL.md` | 'Create the cite-ai-usage skill' | Claude Code skill: citation format and placement rules |
| 2026-10-06 | Claude (claude-opus-5-5) | `.claude/skills/ship/SKILL.md` | 'Create the cite-ai-usage skill' | Citation step now runs `scripts/check_citations.sh` |
| 2026-10-06 | Claude (claude-opus-5-5) | `CLAUDE.md` | 'Write the project CLAUDE.md with the hard rules' | Project rules file for Claude Code: layout and hard rules |
| 2026-10-06 | Claude (claude-opus-5-5) | `common/__init__.py` | 'Build the llm-structured-call skill with its shared helper and tests' | Shared package marker |
| 2026-10-06 | Claude (claude-opus-5-5) | `common/llm_config.py` | 'Build the llm-structured-call skill with its shared helper and tests' | LLM provider profiles (free, paid_dev) and key lookup |
| 2026-10-06 | Claude (claude-opus-5-5) | `common/llm.py` | 'Build the llm-structured-call skill with its shared helper and tests' | Structured LLM helper: strict JSON schema, Pydantic validation, repair retry, provider fallback, call log |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_llm.py` | 'Build the llm-structured-call skill with its shared helper and tests' | Offline tests for every failure path of the helper |
| 2026-10-06 | Claude (claude-opus-5-5) | `pytest.ini` | 'Build the llm-structured-call skill with its shared helper and tests' | Test discovery settings |
| 2026-10-06 | Claude (claude-opus-5-5) | `requirements.txt` | 'Build the llm-structured-call skill with its shared helper and tests' | Dependency minimums |
| 2026-10-06 | Claude (claude-opus-5-5) | `.env.example` | 'Build the llm-structured-call skill with its shared helper and tests' | Template listing the environment variables |
| 2026-10-06 | Claude (claude-opus-5-5) | `.gitignore` | 'Build the llm-structured-call skill with its shared helper and tests' | Ignore rules for secrets, the confidential brief, and dev logs |
| 2026-10-06 | Claude (claude-opus-5-5) | `.claude/skills/llm-structured-call/SKILL.md` | 'Build the llm-structured-call skill with its shared helper and tests' | Claude Code skill: how to add an LLM call |
| 2026-10-06 | Claude (claude-opus-5-5) | `.claude/skills/ship/SKILL.md` | 'Build the llm-structured-call skill with its shared helper and tests' | Bootstrap step now checks the repo's .gitignore |
| 2026-10-06 | Claude (claude-opus-5-5) | `CLAUDE.md` | 'Build the llm-structured-call skill with its shared helper and tests' | Layout adds common/ and tests/; rules 5 and 6 updated for LLM profiles |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/__init__.py` | 'Build the ta-indicators skill with its indicator and signal modules and tests' | Task 1 package marker |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/indicators.py` | 'Build the ta-indicators skill with its indicator and signal modules and tests' | SMA, EMA, Wilder RSI, MACD, Bollinger Bands and volatility from first principles |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/signals.py` | 'Build the ta-indicators skill with its indicator and signal modules and tests' | Rule-based momentum signal: five direction votes, labels, stretch flags |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_indicators.py` | 'Build the ta-indicators skill with its indicator and signal modules and tests' | Reference-value, hand-checked and property tests for the indicators |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_signals.py` | 'Build the ta-indicators skill with its indicator and signal modules and tests' | Scenario tests for every signal label, flag and missing-data path |
| 2026-10-06 | Claude (claude-opus-5-5) | `.claude/skills/ta-indicators/SKILL.md` | 'Build the ta-indicators skill with its indicator and signal modules and tests' | Claude Code skill: how to use, add and test indicators |
| 2026-10-06 | Claude (claude-opus-5-5) | `requirements.txt` | 'Build the ta-indicators skill with its indicator and signal modules and tests' | Adds pandas and numpy |
| 2026-10-06 | Claude (claude-opus-5-5) | `CLAUDE.md` | 'Build the ta-indicators skill with its indicator and signal modules and tests' | Rule 9 points to the ta-indicators skill |
| 2026-10-06 | Claude (claude-opus-5-5) | `.claude/skills/ta-indicators/SKILL.md`, `CLAUDE.md` | 'Trim the lines that quote the brief's mark breakdown' | Removed mark figures from the confidential brief |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/data.py` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | yfinance fetch for a window computed from today, cleaning, retries with backoff, fundamentals |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/summary.py` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | Summary dictionary: price, 52-week range, P/E with fallback, YTD return, indicators, warnings |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/pipeline.py` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | One-call Task 1A entry point composing fetch, indicators, signal and summary |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_data.py` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | Offline tests for the fetch window, cleaning, retries and fallbacks |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_summary.py` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | Hand-checked tests for every summary field and missing-data path |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_pipeline.py` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | Offline end-to-end tests, including Yahoo being unreachable |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_live_yfinance.py` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | Opt-in live check against Yahoo, including the 52-week range against Yahoo's quote |
| 2026-10-06 | Claude (claude-opus-5-5) | `pytest.ini` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | `live` marker, skipped by default |
| 2026-10-06 | Claude (claude-opus-5-5) | `requirements.txt` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | Adds yfinance |
| 2026-10-06 | Claude (claude-opus-5-5) | `common/llm.py` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | `enable_console_logging` also covers task1_financial |
| 2026-10-06 | Claude (claude-opus-5-5) | `.claude/skills/llm-structured-call/SKILL.md`, `.claude/skills/ta-indicators/SKILL.md` | 'Build the Task 1A yfinance data pipeline and summary dictionary' | Point to the shared logging call and to fetch_ohlcv |
| 2026-10-06 | Claude (claude-opus-5-5) | `GLOSSARY.md` | 'Grill the Task 1A news headlines design and record the resolved terms' | Domain glossary: Headline, Publisher, Feed |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/news.py` | 'Build Task 1A step 2, news headlines, as designed in the grilling rounds' | Recent headlines from Yahoo and Google News RSS: recency cutoff, publisher split, de-duplication, retries, clamped count |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/data.py` | 'Build Task 1A step 2, news headlines, as designed in the grilling rounds' | Retry helper made public (`with_retries`) so news.py shares it |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/pipeline.py` | 'Build Task 1A step 2, news headlines, as designed in the grilling rounds' | `run_market_data` returns headlines beside the summary; news warnings join summary['warnings'] |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_news.py` | 'Build Task 1A step 2, news headlines, as designed in the grilling rounds' | Offline tests for parsing, feed order, duplicates, recency, failures and the count limits |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/fixtures/yahoo_rss_sample.xml` | 'Build Task 1A step 2, news headlines, as designed in the grilling rounds' | Hand-written Yahoo-style RSS sample with old, undated and malformed items |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/fixtures/google_news_rss_sample.xml` | 'Build Task 1A step 2, news headlines, as designed in the grilling rounds' | Hand-written Google News-style RSS sample with publisher suffixes and <source> tags |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_pipeline.py` | 'Build Task 1A step 2, news headlines, as designed in the grilling rounds' | News feeds faked in the end-to-end tests; headline assertions |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_live_news.py` | 'Build Task 1A step 2, news headlines, as designed in the grilling rounds' | Opt-in live check: at least 10 recent, unique AAPL headlines |
| 2026-10-06 | Claude (claude-opus-5-5) | `pytest.ini` | 'Build Task 1A step 2, news headlines, as designed in the grilling rounds' | `live` marker description covers the news feeds |
| 2026-10-06 | Claude (claude-opus-5-5) | `GLOSSARY.md` | 'Grill Task 1B (Jev sentiment decisions, LLM text) and record the resolved terms' | Adds Headline sentiment, Sentiment score, Momentum signal, Recommendation |
| 2026-10-06 | Claude (claude-opus-5-5) | `docs/adr/0001-jev-decides-llm-explains.md` | 'Grill Task 1B (Jev sentiment decisions, LLM text) and record the resolved terms' | Decision record: Jev labels headline sentiment, the LLM writes reasons and the Recommendation |
| 2026-10-06 | Claude (claude-opus-5-5) | `common/jev.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Jev client: Choice questions, answer checks (known option, full and normalised probabilities), call logging, no-raise failures |
| 2026-10-06 | Claude (claude-opus-5-5) | `common/llm_config.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | `secret()` lookup shared by the LLM providers and Jev |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/schemas.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Task 1B schemas: headline reason, LLM fallback sentiment, Recommendation with 3-5 sentence validator, records |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/prompts.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Jev headline question and the reason, fallback-sentiment and Recommendation prompts |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/sentiment.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Jev-decides/LLM-explains headline sentiment with fallbacks, and the Sentiment score |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/recommendation.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Computed indicator facts and the LLM Buy/Hold/Sell call with a labelled Hold fallback |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/pipeline.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | `run_analysis()` composing headline sentiment, the Sentiment score and the Recommendation |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/fakes.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Scripted LLM and Jev stand-ins for the Task 1B tests |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_jev.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Offline tests for the Jev client's checks, errors, missing key and logging |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_sentiment.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Offline tests for who decides each headline and the hand-checked Sentiment score |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_recommendation.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Offline tests for sentence counting, the schema, hand-checked facts and the fallback |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_pipeline.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | End-to-end `run_analysis` tests with fake models, including every model failing |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_live_analysis.py` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Opt-in live checks of one Jev request and one LLM Recommendation, skipped without keys |
| 2026-10-06 | Claude (claude-opus-5-5) | `requirements.txt` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Adds typesafe-sdk |
| 2026-10-06 | Claude (claude-opus-5-5) | `.env.example` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | JEV_API_KEY with a comment |
| 2026-10-06 | Claude (claude-opus-5-5) | `CLAUDE.md` | 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds' | Rule 5 records the approved Jev exception; rule 6 and the layout cover common/jev.py |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/report.py` | 'Build the Task 1 notebook and the bonus report as designed in the grilling rounds' | Bonus one-page brief: Markdown template, styled HTML with embedded chart, three-panel indicator chart, top headlines by strength |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_report.py` | 'Build the Task 1 notebook and the bonus report as designed in the grilling rounds' | Offline tests for the brief's sections, top-headline order, placeholders, chart embedding and escaping |
| 2026-10-06 | Claude (claude-opus-5-5) | `task1_financial/task1_equity_research.ipynb` | 'Build the Task 1 notebook and the bonus report as designed in the grilling rounds' | Task 1 Colab notebook: one section per rubric row, independent indicator check, validation demo with a stub client, call-log summary, report |
| 2026-10-06 | Claude (claude-opus-5-5) | `common/jev.py` | 'Build the Task 1 notebook and the bonus report as designed in the grilling rounds' | A missing key is warned about once at start-up, not again for every headline |
| 2026-10-06 | Claude (claude-opus-5-5) | `tests/test_jev.py` | 'Build the Task 1 notebook and the bonus report as designed in the grilling rounds' | Missing-key test checks the single warning |
| 2026-10-06 | Claude (claude-opus-5-5) | `requirements.txt` | 'Build the Task 1 notebook and the bonus report as designed in the grilling rounds' | Adds markdown and matplotlib for the report |
| 2026-10-06 | Claude (claude-opus-5-5) | `CITATIONS.md` | 'Create the cite-ai-usage skill' | This file's structure |

## Adapted open-source code

| Date | Source | File | What was adapted |
|---|---|---|---|
| 2026-10-06 | StockCharts ChartSchool, RSI worked example (cs-rsi.xls), https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-strength-index-rsi | `tests/fixtures/stockcharts_rsi.csv` | QQQQ closes and published 14-day RSI values, extracted verbatim as test reference data |
| 2026-10-06 | StockCharts ChartSchool, moving averages worked example (cs-movavg.xls), https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/moving-averages-simple-and-exponential | `tests/fixtures/stockcharts_movavg.csv` | Intel closes and published 10-day SMA and EMA values, extracted verbatim as test reference data |
| 2026-10-06 | StockCharts ChartSchool, Bollinger Bands worked example (cs-bbands.xls), https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-overlays/bollinger-bands | `tests/fixtures/stockcharts_bbands.csv` | Closes and published Bollinger Band (20, 2) values, extracted verbatim as test reference data |

The conventions in `task1_financial/indicators.py` (SMA-seeded EMA, Wilder
smoothing, population standard deviation for the bands) follow the same
ChartSchool pages and John Bollinger's own definition,
https://www.bollingerbands.com/bollinger-bands. The code itself was written
from those definitions, not copied.

## Teacher-model data generation

None yet.
