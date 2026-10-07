"""QLoRA fine-tuning of the student on a Colab T4, and greedy generation for evaluation.

Every hyperparameter is in HYPERPARAMETERS with the reason it was chosen; the
notebook prints that table, so the values and their reasons live in one place.
torch, transformers and peft are imported inside the functions, because only
the Colab GPU runtime has them; the rest of the package runs without them.
"""
# AI-ASSISTED: Claude (claude-opus-5-5), Prompt: 'Build the Task 2 notebook: QLoRA training, merge, and evaluation against the base model', Date: 2026-10-07

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASE_MODEL = "Qwen/Qwen2.5-Coder-3B-Instruct"
SEED = 42
# Labels at this value are ignored by the loss (the PyTorch cross-entropy convention).
IGNORE_INDEX = -100


@dataclass(frozen=True)
class Hyper:
    value: Any
    reason: str


HYPERPARAMETERS: dict[str, Hyper] = {
    "load_in_4bit / bnb_4bit_quant_type": Hyper(
        "4-bit, nf4",
        "QLoRA's NF4 type is built for normally distributed weights, so it loses less than plain 4-bit; "
        "it brings the 3B model to about 2 GB, leaving the T4's 15 GB for activations.",
    ),
    "bnb_4bit_use_double_quant": Hyper(
        True, "Quantizes the quantization constants too, saving about 0.4 bits per weight at no accuracy cost (QLoRA paper)."
    ),
    "bnb_4bit_compute_dtype": Hyper(
        "float16", "The T4 has no bfloat16 support, so matrix multiplies run in float16, which it accelerates."
    ),
    "lora_r": Hyper(
        16,
        "Learning one fixed docstring layout and a few rules is a narrow skill; rank 16 on every linear layer "
        "gives about 30M trainable weights, enough for that, while rank 64 would mostly memorise ~210 examples.",
    ),
    "lora_alpha": Hyper(
        32,
        "alpha = 2 x r keeps the update scale (alpha / r = 2) fixed if r is changed, the usual pairing in PEFT "
        "examples; with alpha = r the adapter moves half as far per step at the same learning rate.",
    ),
    "lora_dropout": Hyper(
        0.1, "The QLoRA paper used 0.1 for models up to 13B; with this little data some regularisation is worth it."
    ),
    "target_modules": Hyper(
        ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "The QLoRA paper found LoRA on every linear layer is needed to match full fine-tuning; attention-only "
        "adapters underperform. These are all of Qwen2's linear projections.",
    ),
    "learning_rate": Hyper(
        2e-4,
        "The QLoRA paper's rate for 7B-13B models. With only ~45 optimizer steps in total, a smaller rate "
        "would barely move the adapter.",
    ),
    "lr_scheduler_type": Hyper(
        "cosine",
        "Decays smoothly towards zero, so the last epoch takes small steps; on a tiny dataset that protects "
        "the validation loss from late overfitting.",
    ),
    "warmup_ratio": Hyper(
        0.05,
        "The first few steps run at a reduced rate while Adam's moment estimates are still noisy; 5% of ~45 "
        "steps is 3 steps.",
    ),
    "num_train_epochs": Hyper(
        3,
        "One epoch gives a single validation point, so no trend can be shown; beyond three, ~210 examples "
        "start being memorised. The best epoch by validation loss is kept either way.",
    ),
    "per_device_train_batch_size": Hyper(
        4, "The largest batch that fits a T4 at max_seq_length with gradient checkpointing, with headroom."
    ),
    "gradient_accumulation_steps": Hyper(
        4, "Effective batch 4 x 4 = 16: smoother gradients than 4, while still about 15 optimizer steps per epoch."
    ),
    "max_seq_length": Hyper(
        1536,
        "A guard, not a cost: the longest example (system prompt + code + docstring) is under 1,000 tokens, "
        "and batches are padded only to their own longest example. The notebook asserts every example fits, "
        "because truncation would cut the docstring, the only part the loss is computed on.",
    ),
    "loss on": Hyper(
        "the docstring only",
        "Prompt tokens are masked (label -100). The system prompt is the same in every example and the code is "
        "input, so training on them would spend capacity on copying text instead of writing docstrings.",
    ),
    "optim": Hyper(
        "paged_adamw_8bit", "QLoRA's paged optimizer moves optimizer state to CPU on memory spikes instead of crashing with OOM."
    ),
    "gradient_checkpointing": Hyper(
        True, "Recomputes activations in the backward pass: about 30% slower, but it is what makes batch 4 fit on a T4."
    ),
    "fp16": Hyper(True, "Mixed precision training in float16, matching the compute dtype the T4 supports."),
    "seed": Hyper(SEED, "Fixes data order, LoRA initialisation and dropout, so the run can be repeated."),
    "max_new_tokens": Hyper(
        384, "The longest reference docstring is under 250 tokens (checked in the notebook), so no answer is cut short."
    ),
    "decoding": Hyper(
        "greedy", "Deterministic answers, so the base and fine-tuned scores do not depend on sampling luck."
    ),
}


def hp(name: str) -> Any:
    return HYPERPARAMETERS[name].value


def load_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prompt_text(messages: list[dict[str, str]], tokenizer) -> str:
    """The system and user turns in the model's chat template, ending where the answer starts."""
    return tokenizer.apply_chat_template(messages[:2], tokenize=False, add_generation_prompt=True)


def tokenize_record(record: dict[str, Any], tokenizer) -> dict[str, list[int]]:
    """input_ids for the whole conversation, with labels masked everywhere except the answer.

    Both texts come from the tokenizer's own chat template, so the prompt is an exact
    prefix of the full conversation; the answer includes <|im_end|>, so the model also
    learns where to stop.
    """
    prompt = prompt_text(record["messages"], tokenizer)
    full = tokenizer.apply_chat_template(record["messages"], tokenize=False)
    if not full.startswith(prompt):
        raise ValueError(f"{record['id']}: the prompt is not a prefix of the conversation")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"{record['id']}: prompt tokens differ inside the full conversation")
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :],
    }


def load_quantized_model():
    """The student in 4-bit NF4, prepared for k-bit training, with LoRA adapters attached."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=hp("bnb_4bit_use_double_quant"),
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, quantization_config=quant, device_map={"": 0})
    model.config.use_cache = False  # the generation cache is useless in training and clashes with checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=hp("gradient_checkpointing"))
    lora = LoraConfig(
        r=hp("lora_r"),
        lora_alpha=hp("lora_alpha"),
        lora_dropout=hp("lora_dropout"),
        target_modules=hp("target_modules"),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def build_trainer(model, tokenizer, train_rows: list[dict], val_rows: list[dict], output_dir: Path):
    """A plain transformers Trainer: evaluates, logs and saves once per epoch, keeps the best epoch."""
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments

    batch = hp("per_device_train_batch_size") * hp("gradient_accumulation_steps")
    total_steps = math.ceil(len(train_rows) / batch) * hp("num_train_epochs")
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hp("num_train_epochs"),
        per_device_train_batch_size=hp("per_device_train_batch_size"),
        per_device_eval_batch_size=hp("per_device_train_batch_size"),
        gradient_accumulation_steps=hp("gradient_accumulation_steps"),
        learning_rate=hp("learning_rate"),
        lr_scheduler_type=hp("lr_scheduler_type"),
        warmup_steps=max(1, math.ceil(total_steps * hp("warmup_ratio"))),
        optim=hp("optim"),
        fp16=hp("fp16"),
        gradient_checkpointing=hp("gradient_checkpointing"),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=SEED,
    )
    # Pads input_ids with the pad token and labels with -100, so padding never counts in the loss.
    collator = DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=IGNORE_INDEX)
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_rows,
        eval_dataset=val_rows,
        data_collator=collator,
        processing_class=tokenizer,
    )


def epoch_losses(log_history: list[dict[str, Any]]) -> list[dict[str, float]]:
    """One row per epoch: the mean training loss over the epoch and the validation loss after it."""
    train = {round(e["epoch"]): e["loss"] for e in log_history if "loss" in e and "epoch" in e}
    val = {round(e["epoch"]): e["eval_loss"] for e in log_history if "eval_loss" in e}
    return [{"epoch": n, "train_loss": train.get(n), "val_loss": val[n]} for n in sorted(val)]


def load_fp16(model_id_or_dir: str | Path):
    """A model in float16 on the GPU: the base model for the baseline, or the merged model."""
    import torch
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(str(model_id_or_dir), dtype=torch.float16, device_map={"": 0})


def merge_adapter(adapter_dir: Path):
    """The base model in float16 with the LoRA adapter folded into its weights.

    Merging into the float16 base, not the 4-bit training copy, avoids baking
    quantization error into the saved model.
    """
    from peft import PeftModel

    return PeftModel.from_pretrained(load_fp16(BASE_MODEL), str(adapter_dir)).merge_and_unload()


def generate(model, tokenizer, records: list[dict[str, Any]], batch_size: int = 8) -> list[str]:
    """Greedy answers for each record's system + user turns, decoded without special tokens."""
    import torch

    tokenizer.padding_side = "left"  # a decoder generates from the right end, so pad on the left
    answers: list[str] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        chunk = records[start : start + batch_size]
        texts = [prompt_text(r["messages"], tokenizer) for r in chunk]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=hp("max_new_tokens"),
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        answers += tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
    tokenizer.padding_side = "right"
    return [a.strip() for a in answers]


def free_gpu() -> None:
    """Empty the CUDA cache after the caller has `del`eted its models, so the next model has the whole T4."""
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
