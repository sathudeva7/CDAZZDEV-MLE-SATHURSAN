<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write the root README and a Task 1 REFLECTION as decided in the grilling round', Date: 2026-10-07 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'yes write the reflection (rewrite it to cover Tasks 1 and 3)', Date: 2026-10-07 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'can u add reflectionmd file now (add Task 2 within 600 words)', Date: 2026-10-07 -->
# Reflection

## Architectural decisions

**Code computes, the LLM writes.** Models reason well over relationships but get
arithmetic wrong, so every number is computed in code. Task 1's Buy/Hold/Sell prompt
gets computed facts (distance from each moving average, whether RSI is rising), not
my rule-based momentum label. Task 3 computes hedge levels. The LLM picks a strategy,
and code fills in prices and drops evidence citing a tool that returned nothing usable.

**One validated path for every model call.** Each call uses a prompt constant, a
strict JSON schema, Pydantic validation, one repair retry, provider fallback, a
fallback value and a JSONL log. Headline sentiment comes from Jev, a paid decision
model (a deliberate exception to the free-tier setup), because it returns
probabilities rather than a self-reported confidence.

**A hand-written LangGraph loop.** My own tools node enforces the budget (8 calls,
6 turns), refuses tools the agent was not given, and turns failures into
observations. Tools never raise. Each returns `{status, data, error, hint}`, so when
`get_news` fails the agent follows the hint to `web_search`. Agent A hands Agent B a
Pydantic `DataBrief`, and the critique is a fixed graph edge, so it runs exactly once.

**Honest fallbacks.** Every failure becomes a logged warning, so runs finish. With no
price data (a bad ticker), the report comes from a template, not the LLM, which would
invent risks.

**Task 2: fine-tune for behaviour, not knowledge.** I rejected compliance Q&A
because retrieval solves it. A docstring's answer is in the code, so what
fine-tuning teaches is layout and discipline. One `ast` check defines correctness:
it filters the teacher's labels and scores both models. The teacher writes each
docstring from the code alone, so labels describe the code as written, and the loss
covers only docstring tokens. The teacher was GPT-6.1 Sol, a second deliberate
exception: free Groq would have taken three days and shared its daily budget with
the Task 1 and 3 runs. It cost $1.90.

## What I would improve with more time

- Check evidence values, not only their source: every number an LLM quotes should
  match the tool output it cites.
- Price hedges from option chains and implied volatility, not historical volatility.
- Evaluate Tasks 1 and 3: backtest Recommendations against forward returns, and
  score reports against a rubric.
- Task 2's errors moved from structure into prose. The `ast` check flagged 1 of 26
  answers, but my review of 12 found 4 hallucinated (wrong exception names, reversed
  conditions). Next: pair each `raise` with its condition in the check, run a DPO
  pass on correct-versus-hallucinated pairs, and regenerate when the check fails.
- For production: an endpoint that returns 202 with a job id, a worker for slow
  agent runs, and a persistent checkpointer.

## Limitations

- Groq's free tier allows 8K tokens a minute, so agents read compact digests and
  runs wait on rate limits. A paid model I compared during development batched tool
  calls and exposed a hedge-level gap I then fixed.
- Tool order varies between runs. The trace, not the code, records what happened.
- Only headline titles are read, so near-duplicate insider-sale headlines can pull
  the Sentiment score toward one story.
- Task 2's references are one teacher's wording, checked by `ast` but not by a
  person, so ROUGE-L partly rewards the teacher's style. The test set has only 26
  functions, and training peaked at 14.3 of the T4's 14.5 GB.
- No ground truth says whether a call or a risk was right. Outputs can be checked
  for reasoning and evidence, not accuracy.
