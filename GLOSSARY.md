<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Grill the Task 1A news headlines design and record the resolved terms', Date: 2026-10-06 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Grill Task 1B (Jev sentiment decisions, LLM text) and record the resolved terms', Date: 2026-10-06 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 1: the five agent tools, their tests and the new-tool skill, as designed in the grilling rounds', Date: 2026-10-07 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 2: the 3A agent loop, report, hedge levels, printer and short-term memory, as designed in the grilling rounds', Date: 2026-10-07 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 3 PR 3: the 3B two-agent pipeline with the critique loop and the persistent cache, as designed in the grilling rounds', Date: 2026-10-07 -->
# Equity research

The words this repo uses for the market data, news and analysis behind a
first-pass equity research brief on one ticker.

## News

**Headline**:
The title of one recent news article about the ticker, with when it was
published and where it was found.
_Avoid_: Article, story, news item

**Publisher**:
The news outlet that wrote the article, such as Reuters or Bloomberg.
_Avoid_: Source, outlet, provider

**Feed**:
The public RSS listing a headline was collected from, such as Yahoo Finance
or Google News. It is not the outlet that wrote the article.
_Avoid_: Source, provider, endpoint

## Analysis

**Headline sentiment**:
Whether one headline is likely to lift (positive), weigh on (negative) or
not clearly move (neutral) the ticker's share price, with how sure that
judgement is. It is about the share price, not the tone of the wording.
_Avoid_: Tone, mood, polarity

**Sentiment score**:
One number from -1 to +1 summarising the headline sentiments for a ticker:
-1 is uniformly negative news, +1 uniformly positive, 0 balanced or no news
that matters.
_Avoid_: Overall sentiment, news score

**Momentum signal**:
The rule-based reading of the latest technical indicators, from Strong
Bearish to Strong Bullish. No model is involved.
_Avoid_: Signal, technical signal, trend signal

**Recommendation**:
The LLM's Buy, Hold or Sell call for the ticker, with a justification that
reasons over the indicators and the sentiment score together.
_Avoid_: Signal, rating, verdict, trade signal

## Agents

**Tool result**:
What one agent tool call returns: a status (ok, empty or error), the tool's
data, and, on failure, a hint naming the other tools that could fill the gap.
_Avoid_: Tool output, response, observation

**Digest**:
The short JSON view of a tool result that the agent's model reads; the full
data stays in the tool result for code to use.
_Avoid_: Summary, observation, tool message

**Observation**:
One tool call as the agent loop records it: the tool, its arguments, the
agent's stated reason (`why`), the status, the digest the agent read, and the
full tool result. The report is written from the observations alone.
_Avoid_: Step, event, tool message

**Expected move**:
The one-standard-deviation price change over the hedge horizon: price x
volatility x sqrt(63 / 252) for 90 days. About two times in three the price
ends within one expected move of today's.
_Avoid_: Range, volatility band

**Agent trace**:
The `agent_trace.jsonl` log: one line per tool call (and per handoff,
critique and cache lookup) with its inputs, digest cut to 200 characters,
status and duration.
_Avoid_: Log, call log (that is the LLM's `llm_calls.jsonl`)

**Data brief**:
Agent A's typed handoff to Agent B: price, volatility and sentiment figures
copied from A's tool results, the hedge levels, and A's key observations.
_Avoid_: Summary, A's output, report

**Clarification request**:
Agent B's one question back to Agent A, limited to data A's tools can
supply (price data, volatility or sentiment), with the reason it matters.
_Avoid_: Critique, follow-up, feedback

**Cached run**:
A finished 3B result saved as JSON under the ticker and the New York market
date; a later run that day loads it instead of calling any tool.
_Avoid_: Memory, snapshot, saved state
