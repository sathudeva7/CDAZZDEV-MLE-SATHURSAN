<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write the root README and a Task 1 REFLECTION as decided in the grilling round', Date: 2026-10-07 -->
# Reflection

## Architectural decisions

**A decision model labels headlines, and the LLM explains them.** An LLM's
"confidence" is a number it writes about itself. I used Jev, TypeSafe's
decision model, to choose each headline's sentiment, because it returns a
probability for every label. The headline score is then p(positive) −
p(negative), and the Sentiment score is the mean of those. The LLM writes the
reason for a label it was given, so the two cannot contradict each other. Jev
is a paid API, so it is optional: without its key, the LLM labels headlines
itself and the run stays on free tiers.

**Facts, not a verdict.** The Buy/Hold/Sell prompt receives relationships I
compute in code, such as distance from each moving average, whether the
50/200-day gap is widening, and which way RSI and the MACD histogram are moving.
It does not receive my rule-based momentum label, so the model has to weigh the
indicators against each other instead of copying a conclusion. Models reason
well over relationships but get arithmetic wrong, so the arithmetic stays in code.

**One validated path for every model call.** Each call uses a prompt constant,
a strict JSON schema, Pydantic validation, one repair retry with the error,
provider fallback, and a caller-supplied fallback value, and is logged to
JSONL. The rest of the code never handles a raw model response.

**Warnings instead of exceptions.** Every data problem becomes a logged warning
and a placeholder, so the notebook always finishes and reports what it could not do.

## What I would improve with more time

- Group near-duplicate headlines by meaning, not exact title.
- Evaluate the outputs: backtest the Recommendation against forward returns, and
  check sentiment labels against a hand-labelled set.
- Score headlines concurrently. The 15 calls currently run one after another.
- For production: an endpoint that returns 202 with a job id, and a worker that
  runs the slow model calls.

## Limitations

- Several near-identical insider-sale headlines can pull the Sentiment score
  toward one story. They partly explain the negative score in the submitted run.
- Only headline titles are read, not article text.
- There is no ground truth here for whether a Buy/Hold/Sell call was right; the
  justification can be checked for reasoning, not for accuracy.
- Free tiers rate-limit: Yahoo's news feed returned HTTP 429 during testing, and
  the pipeline fell back to Google News.
