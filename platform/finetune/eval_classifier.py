#!/usr/bin/env python3
"""Evaluate fine-tuned LoRA classifier on held-out val split.

Usage:
  python platform/finetune/eval_classifier.py
  python platform/finetune/eval_classifier.py --val-split path/to/val.json
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

DEFAULT_CHECKPOINT = FINETUNE_DIR / "checkpoints" / "flan-t5-lora-rights-tier"
DEFAULT_VAL = DEFAULT_CHECKPOINT / "val_split.json"
DEFAULT_OUTPUT = DEFAULT_CHECKPOINT / "eval_results.json"

from config import LABELS  # noqa: E402
from inference import load_finetuned, predict_batch  # noqa: E402


def load_val_rows(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected JSON array in {path}")
    return [row for row in data if row.get("label") in LABELS]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0)
    weighted_f1 = f1_score(
        y_true, y_pred, labels=list(LABELS), average="weighted", zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(LABELS),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(LABELS))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": {label: report.get(label, {}) for label in LABELS},
        "confusion_matrix": {
            "labels": list(LABELS),
            "rows": matrix.tolist(),
        },
    }


def evaluate(
    *,
    adapter_dir: Path,
    val_path: Path,
    output_path: Path,
) -> dict:
    rows = load_val_rows(val_path)
    if not rows:
        raise SystemExit(f"No labeled rows in {val_path}")

    model, tokenizer, device = load_finetuned(adapter_dir)
    texts = [row["text"] for row in rows]
    y_true = [row["label"] for row in rows]
    y_pred, total_latency = predict_batch(texts, model, tokenizer, device)

    metrics = compute_metrics(y_true, y_pred)
    errors = [
        {
            "id": row.get("id"),
            "text": row["text"][:120],
            "expected": row["label"],
            "predicted": pred,
        }
        for row, pred in zip(rows, y_pred)
        if row["label"] != pred
    ]

    result = {
        "adapter_dir": str(adapter_dir),
        "val_split": str(val_path),
        "n_samples": len(rows),
        "device": device,
        "latency_ms_total": round(total_latency * 1000, 2),
        "latency_ms_per_snippet": round((total_latency / len(rows)) * 1000, 2),
        **metrics,
        "errors": errors,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Samples: {len(rows)}  Device: {device}")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Macro F1:  {metrics['macro_f1']:.3f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
    print(f"Latency:   {result['latency_ms_per_snippet']:.1f} ms/snippet")
    print("\nPer class:")
    for label in LABELS:
        stats = metrics["per_class"].get(label, {})
        print(
            f"  {label:8} P={stats.get('precision', 0):.2f} "
            f"R={stats.get('recall', 0):.2f} F1={stats.get('f1-score', 0):.2f} "
            f"(n={stats.get('support', 0)})"
        )
    if errors:
        print(f"\nMisclassified ({len(errors)}):")
        for err in errors[:10]:
            print(f"  expected={err['expected']!r} predicted={err['predicted']!r}")
            print(f"    {err['text']!r}")
    print(f"\nWrote → {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LoRA rights tier classifier")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT / "adapter",
    )
    parser.add_argument("--val-split", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        raise SystemExit(
            f"Adapter not found: {args.adapter_dir}\n"
            "Run: python platform/finetune/train_classifier.py"
        )
    if not args.val_split.exists():
        raise SystemExit(f"Val split not found: {args.val_split}")

    evaluate(adapter_dir=args.adapter_dir, val_path=args.val_split, output_path=args.output)


if __name__ == "__main__":
    main()
