<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 notebook: QLoRA training, merge, and evaluation against the base model', Date: 2026-10-07 -->
<!-- AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Write 2B.6 and 2C.7 from the submission run's outputs', Date: 2026-10-07 -->
# Task 2: fine-tuning a small code model to write docstrings

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sathudeva7/CDAZZDEV-MLE-SATHURSAN/blob/main/task2_genai/task2_docstring_finetune.ipynb)

`Qwen/Qwen2.5-Coder-3B-Instruct` is fine-tuned with QLoRA on Colab's free T4 to write a
Google-style docstring for one Python function. The answer comes from the code, so
retrieval has nothing to add. What fine-tuning changes is behaviour: one fixed layout,
every parameter and no others, `Yields` for generators, and only the exceptions the
code really raises.

A docstring is checked against the function with Python's `ast` module
(`docstrings.py`), which labels it **correct**, **partial** (something missing, nothing
invented) or **hallucinated** (a parameter, exception or section the code does not
have). The same check filters the teacher's training labels and scores both models.

**Merged model:** https://huggingface.co/sathudeva7/qwen2.5-coder-3b-docstrings (public)

## Results (submission run, Colab T4, 66 min)

All figures are from the 26 held-out test functions, with the same system prompt and greedy
decoding for both models.

| Metric | Base | Fine-tuned |
|---|---|---|
| ROUGE-L F1 (mean) | 0.242 | **0.540** |
| BERTScore F1 (mean, rescaled) | 0.035 | **0.594** |
| `ast` check: correct / partial / hallucinated | 0 / 26 / 0 | **22 / 3 / 1** |

- **Validation loss per epoch:** 0.989 → 0.903 → 0.894.
- **Manual review of 12 fine-tuned answers:** 6 correct, 2 partial, 4 hallucinated, a
  **33% hallucination rate**.
- **What changed:** fine-tuning fixed the structure. The base model mostly wrote one prose
  paragraph, with `Args`, `Raises` and `Yields` almost always missing. The remaining errors
  are in the prose: wrong exception names, reversed conditions and invented behaviour.
  Section 2C.7 of the notebook has the examples and the next steps.

## Pipeline

1. **Data** (`generate_data.py`, run once locally, output committed in `data/`).
   - A recipe grid of 12 topics × 4 kinds (function, method, generator, async) ×
     3 levels (`recipes.py`) fixes 288 recipe ids in advance.
   - Teacher call 1 writes new functions, and each is validated.
   - Teacher call 2 sees only the code and writes its docstring.
   - Only docstrings the `ast` check labels `correct` are kept, then split 80/10/10
     by kind.
   - The teacher is GPT-6.1 Sol, a paid API and a deliberate exception to the
     free-tier setup (`docs/adr/0002-paid-teacher-for-task2.md`). Its prompts are in
     `prompts.py` and its calls in `logs/teacher_calls.jsonl`.
2. **Training** (`finetune.py`, in the notebook).
   - 4-bit NF4 with double quantization and LoRA r=16 on every linear layer.
   - The loss covers the docstring tokens only.
   - 3 epochs, with validation loss after each.
   - The adapter is merged with `merge_and_unload()` into the float16 base and
     pushed to the Hugging Face Hub.
   - Every hyperparameter's reason is in `finetune.HYPERPARAMETERS`.
3. **Evaluation** (`metrics.py`, in the notebook).
   - ROUGE-L and BERTScore F1 for the base and fine-tuned models on the same
     test set.
   - The `ast` label of every answer.
   - A manual review of 12 fine-tuned answers, giving the hallucination rate.

| File | What it holds |
|---|---|
| `task2_docstring_finetune.ipynb` | The Colab notebook: data report, training, merge, evaluation |
| `data/train.jsonl`, `val.jsonl`, `test.jsonl` | The chat-format dataset (system, user, assistant turns) |
| `data/functions.jsonl`, `labels.jsonl`, `rejected.jsonl` | Every teacher output, with validation and check results |
| `evaluation/` | Run outputs: `results.json`, `predictions.jsonl` (both models' answers with scores), loss history and plot |
| `logs/teacher_calls.jsonl` | One line per teacher call: prompt, model, tokens, outcome |

## Running

- **Notebook:** open it with the badge, choose Runtime → Change runtime type → T4 GPU,
  and add an `HF_TOKEN` with write access under Colab's Secrets to publish the
  merged model. Then Runtime → Run all.
- **Regenerating the data** needs `OPENAI_API_KEY` in `.env`:
  `python -m task2_genai.generate_data`. The script resumes where it stopped and
  skips recipe ids it already has.
  `python -m task2_genai.generate_data --split` only rebuilds the splits.
- **Tests:** `pytest tests/test_task2_data.py` runs offline, with no GPU and no key.
