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
