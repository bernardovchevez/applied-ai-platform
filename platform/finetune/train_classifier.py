#!/usr/bin/env python3
"""LoRA fine-tune flan-t5-base for rights tier classification.

Usage:
  pip install -r requirements-finetune.txt
  python platform/finetune/generate_dataset.py
  python platform/finetune/train_classifier.py
  python platform/finetune/train_classifier.py --epochs 1 --max-steps 10  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
LAB_ROOT = FINETUNE_DIR.parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(FINETUNE_DIR))

DEFAULT_DATASET = FINETUNE_DIR / "dataset" / "rights_tier.jsonl"
DEFAULT_OUTPUT = FINETUNE_DIR / "checkpoints" / "flan-t5-lora-rights-tier"

from config import (  # noqa: E402
    LABELS,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_INPUT_LENGTH,
    MAX_LABEL_LENGTH,
    MODEL_ID,
    TASK_PREFIX,
)


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def stratified_split(rows: list[dict], val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    from sklearn.model_selection import train_test_split

    labels = [row["label"] for row in rows]
    train, val = train_test_split(
        rows,
        test_size=val_ratio,
        random_state=seed,
        stratify=labels,
    )
    return train, val


def build_preprocess_fn(tokenizer):
    prefix = TASK_PREFIX

    def preprocess(batch: dict) -> dict:
        inputs = [f"{prefix} {text}" for text in batch["text"]]
        model_inputs = tokenizer(
            inputs,
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
        )
        labels = tokenizer(
            batch["label"],
            max_length=MAX_LABEL_LENGTH,
            truncation=True,
        )
        model_inputs["labels"] = [
            [(tok if tok != tokenizer.pad_token_id else -100) for tok in seq]
            for seq in labels["input_ids"]
        ]
        return model_inputs

    return preprocess


def train(
    *,
    dataset_path: Path,
    output_dir: Path,
    val_ratio: float,
    seed: int,
    epochs: float,
    batch_size: int,
    learning_rate: float,
    max_steps: int,
) -> dict:
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    device = pick_device()
    print(f"Device: {device}")
    print(f"Model: {MODEL_ID}")
    print(f"Dataset: {dataset_path}")

    rows = load_jsonl(dataset_path)
    rows = [r for r in rows if r.get("label") in LABELS]
    if len(rows) < 20:
        raise SystemExit(f"Need at least 20 labeled rows; got {len(rows)}")

    train_rows, val_rows = stratified_split(rows, val_ratio=val_ratio, seed=seed)
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}")

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(LORA_TARGET_MODULES),
        inference_mode=False,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    preprocess = build_preprocess_fn(tokenizer)
    train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
    val_ds = val_ds.map(preprocess, batched=True, remove_columns=val_ds.column_names)

    use_fp16 = device == "cuda"
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch" if max_steps <= 0 else "steps",
        eval_steps=50 if max_steps > 0 else None,
        save_strategy="epoch" if max_steps <= 0 else "steps",
        save_steps=50 if max_steps > 0 else None,
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=MAX_LABEL_LENGTH,
        load_best_model_at_end=max_steps <= 0,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        report_to="none",
        fp16=use_fp16,
        dataloader_pin_memory=device == "cuda",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    train_result = trainer.train()
    metrics = trainer.evaluate()
    print(f"Eval loss: {metrics.get('eval_loss')}")

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    manifest = {
        "model_id": MODEL_ID,
        "labels": list(LABELS),
        "task_prefix": TASK_PREFIX,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
        "device": device,
        "train_loss": train_result.training_loss,
        "eval_loss": metrics.get("eval_loss"),
        "lora": {
            "r": LORA_R,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "target_modules": list(LORA_TARGET_MODULES),
        },
    }
    (output_dir / "train_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "val_split.json").write_text(
        json.dumps(val_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Adapter saved → {adapter_dir}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune rights tier classifier")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, cap training steps (smoke test)",
    )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(
            f"Dataset not found: {args.dataset}\n"
            "Run: python platform/finetune/generate_dataset.py"
        )

    train(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
