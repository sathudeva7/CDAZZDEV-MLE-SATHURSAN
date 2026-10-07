<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write the project CLAUDE.md with the hard rules', Date: 2026-10-06 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build Task 1B (Jev headline sentiment, LLM reasons and Recommendation) as designed in the grilling rounds', Date: 2026-10-06 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write the root README and a Task 1 REFLECTION as decided in the grilling round', Date: 2026-10-07 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Add a stronger paid OpenAI model (gpt-6.1-sol) for testing, in a separate file so it can be deleted before submission', Date: 2026-10-07 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 data script as designed in the grilling rounds', Date: 2026-10-07 -->
# CDAZZDEV Senior MLE assessment

This repo is a take-home assessment submission, published as the public GitHub
repo `CDAZZDEV-MLE-<Name>`. Reviewers grade from the code, the **executed**
Colab notebook outputs, `CITATIONS.md` and `REFLECTION.md`. Then the candidate
has to defend every part in an interview. Write for that reader: readable,
commented code where every choice can be explained.

The brief is `tasks.md`. Before working on a task, read its section there,
because the rubric tables list exactly what earns marks. Tasks 1 and 3 are
built; current focus: Task 2 (Generative AI), due 2026-10-08.

## Layout

- `task1_financial/`, `task3_agentic/`, and `task2_genai/` if attempted: each
  holds one Colab notebook, the task's modules, and a README with a Colab badge.
- Task 3's tools import Task 1's pipeline (prices, indicators, news, sentiment)
  so the logic lives in one place.
- `common/` holds code shared by every task: the structured LLM helper
  (`llm.py`), its provider profiles (`llm_config.py`), and the Jev client
  (`jev.py`).
- `tests/` holds offline pytest tests, and `scripts/` holds the repo gates:
  `check_secrets.sh` and `check_citations.sh`.

## Hard rules

Breaking a rule marked (DQ) disqualifies the whole submission.

1. **Credentials live in the environment only (DQ).** Read keys from
   `os.environ` locally, loaded from a gitignored `.env`, and from
   `google.colab.userdata` in notebooks. `scripts/check_secrets.sh` must
   print `clean` before every commit.
2. **Notebooks ship executed (DQ).** Run each notebook top to bottom and commit
   it with every cell output visible.
3. **The brief stays private.** `tasks.md` and `tasks.pdf` are confidential, so
   keep them out of every commit.
4. **Cite in the same turn as the change.** Every file Claude writes or edits
   gets its marker and its `CITATIONS.md` row, following the `cite-ai-usage`
   skill, and `scripts/check_citations.sh` exits 0.
5. **Submitted runs are free tier.** Notebooks are executed for submission on
   the `free` LLM profile, with Groq, OpenRouter free models, yfinance,
   duckduckgo-search and Colab's free GPU. The `paid_dev` profile is for local
   testing only. <!-- TESTING-ONLY(openai) --> So is `openai_dev` (OpenAI's
   gpt-6.1-sol, `common/llm_openai.py`), added 2026-10-07 to judge Task 3's
   output; delete it and every `TESTING-ONLY(openai)` line before submission,
   except the Sol provider path Task 2's teacher uses (see below): retag those
   lines for Task 2 instead of deleting them.
   Ask before adding anything else that needs a paid plan.
   One deliberate exception (2026-10-06), the candidate's own choice: Jev,
   TypeSafe's paid decision model, labels headline sentiment in every profile
   when its key is set (`docs/adr/0001`). A second one (2026-10-07), also the
   candidate's choice: GPT-6.1 Sol is Task 2's teacher, run once locally by
   `task2_genai/generate_data.py`, with the data committed (`docs/adr/0002`).
   Public docs call each "a deliberate exception to the free-tier setup",
   never "approved".
6. **LLM calls go through `common/llm.py`.** Follow the `llm-structured-call`
   skill: `Prompt` constants with separate system and user messages, a
   Pydantic schema, and a fallback value, so the caller always gets a result.
   Jev requests go through `common/jev.py` the same way, with `JevQuestion`
   constants, and the LLM writes every piece of text Jev cannot.
7. **Failures degrade gracefully.** Missing data, empty API results and tool
   errors each produce a logged fallback, so the pipeline and the agents run to
   completion.
8. **Named constants and computed dates.** Windows, periods and thresholds are
   named constants (`RSI_PERIOD = 14`). Date ranges are computed from today's
   date, never written as literal date strings.
9. **Indicators from first principles.** Write SMA, RSI, MACD and Bollinger
   Bands with pandas and numpy only, following the `ta-indicators` skill.
   TA-Lib and indicator libraries forfeit the indicator-accuracy criterion.
