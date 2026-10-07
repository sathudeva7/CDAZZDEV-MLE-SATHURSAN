<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write the root README and a Task 1 REFLECTION as decided in the grilling round', Date: 2026-10-07 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'yes write the reflection (rewrite it to cover Tasks 1 and 3)', Date: 2026-10-07 -->
# Reflection

## Architectural decisions

**Code computes, the LLM writes.** Models reason well over relationships but
get arithmetic wrong, so every number is computed in code. Task 1's
Buy/Hold/Sell prompt gets computed facts (distance from each moving average,
whether RSI is rising), not my rule-based momentum label. Task 3's hedge
levels (the expected 90-day move and candidate strikes) are computed, and the
LLM picks a strategy and names levels. Code then fills in the prices and drops
evidence that cites a tool which returned nothing usable.

**One validated path for every model call.** Each call uses a prompt constant,
a strict JSON schema, Pydantic validation, one repair retry, provider fallback,
a fallback value and a JSONL log. Headline sentiment comes from Jev, a paid
decision model (a deliberate exception to the free-tier setup), because it
returns probabilities rather than a self-reported confidence. Without its key,
the LLM labels headlines itself.

**A hand-written LangGraph loop.** I wrote the tools node myself rather than
using the prebuilt one, so one place enforces the budget (8 calls, 6 turns),
refuses a tool the agent was not given, and turns every failure into an
observation. Tools never raise: each returns `{status, data, error, hint}`,
and the hint names the alternative. When `get_news` fails, the agent reads
the hint and switches to `web_search`. Tool restriction in 3B is enforced by
binding, not by the prompt.

**Typed coordination.** Agent A hands Agent B a Pydantic `DataBrief`. The
critique (one `ClarificationRequest` and its typed answer) is a fixed graph
edge, so it runs exactly once. If A does not fetch the requested data, the
pipeline does and labels the answer. Cache files are validated on load, and a
report written from the template is never cached.

**Honest fallbacks.** Every failure becomes a logged warning, so runs always
finish. When no price data came back (a bad ticker), the report is written
from a template rather than by the LLM, which would otherwise invent risks.

## What I would improve with more time

- Check evidence values, not only their source: every number an LLM quotes
  should match the tool output it cites.
- Price the hedges from option chains and implied volatility. Today's levels
  use historical volatility and say premiums are unpriced.
- Evaluate outputs: backtest Recommendations against forward returns, score
  reports against a rubric, and check sentiment against a hand-labelled set.
- Group near-duplicate headlines by meaning, not exact title.
- For production: an endpoint that returns 202 with a job id, a worker for the
  slow agent runs, and a persistent checkpointer so memory outlives a session.

## Limitations

- Groq's free tier allows 8K tokens a minute. Agents therefore read compact
  digests, not raw tool output, and runs spend time waiting on rate limits.
  During development I compared a stronger paid model (gpt-6.1-sol): it batched
  tool calls and wrote better-evidenced reports, and its clarification request
  (horizon-matched volatility) exposed the hedge-level gap I then fixed.
- Tool order varies between runs, because the model decides it. The trace,
  not the code, records what happened.
- Only headline titles are read; near-identical insider-sale headlines can
  pull the Sentiment score toward one story.
- There is no ground truth for whether a call or a risk was right. Outputs can
  be checked for reasoning and evidence, not accuracy.
