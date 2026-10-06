<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Grill Task 1B (Jev sentiment decisions, LLM text) and record the resolved terms', Date: 2026-10-06 -->
# Jev decides headline sentiment, the LLM writes the text

Each headline's sentiment label comes from Jev, TypeSafe's decision model. Jev
returns a probability for every option, so the per-headline score
p(positive) − p(negative) and the confidence are computed from a distribution.
An LLM's confidence is only a number it writes about itself. Jev cannot
generate text, so the LLM writes each `brief_reason`, explaining Jev's label
without changing it.

## Considered Options

- **LLM only:** simpler, but every label would come with a self-reported
  confidence and no probabilities, so the Sentiment score would rest on
  numbers nobody measured.
- **Jev also picks Buy/Hold/Sell:** rejected. The Recommendation must be the
  LLM's own reasoning over indicator combinations. A justification written for
  a decision made elsewhere would only rationalise that decision.

## Consequences

- If Jev is unreachable, the LLM labels that headline itself and the result is
  marked `scored_by="llm"`, so the pipeline still finishes.
- Jev requests go to the same call log as LLM calls, so one file shows every
  model decision.
