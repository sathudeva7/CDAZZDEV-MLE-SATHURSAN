<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 data script as designed in the grilling rounds', Date: 2026-10-07 -->
# GPT-6.1 Sol writes Task 2's training data

Task 2's teacher is OpenAI's GPT-6.1 Sol, a paid API. It is a deliberate exception to
the free-tier setup, the second after Jev (ADR 0001). The candidate chose it on
2026-10-07, the day before submission.

Why:

- **Time.** Groq's free tier allows 200,000 tokens a day for `openai/gpt-oss-120b`,
  shared across the account. Two teacher calls per example need about 2,700 tokens,
  so about 70 examples a day. The dataset would have taken three days.
- **The budget is shared.** The Task 1 and Task 3 submission runs need about 170,000
  of the same 200,000 free tokens on the day they run.
- **Cost.** The whole dataset cost a few dollars: Sol at $2 per million input tokens
  and $10 per million output tokens, about 290 short calls.

## Considered Options

- **Groq free (`gpt-oss-120b`):** the same quality class and free, but three days of
  generation, and it would compete with the Task 1 and Task 3 runs for one daily budget.
- **OpenRouter free models:** capped at 50 requests a day, far too few for about 330 calls.

## Consequences

- The data is generated once, locally, by `task2_genai/generate_data.py`, and committed.
  The notebook only reads it, so a reviewer re-runs training and evaluation for free.
  Regenerating the data needs an `OPENAI_API_KEY`.
- Every teacher call is logged in `task2_genai/logs/teacher_calls.jsonl`, with tokens.
- The teacher (OpenAI) and the student (Qwen) are different model families, as the
  brief requires.
- The `openai_dev` profile that Task 2's teacher is built from stays in the repo after
  the Task 3 testing code is removed.
