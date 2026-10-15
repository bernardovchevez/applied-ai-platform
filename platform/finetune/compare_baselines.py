#!/usr/bin/env python3
"""Compare fine-tuned LoRA vs zero-shot Ollama on val split.

Usage:
  python platform/finetune/compare_baselines.py
  python platform/finetune/compare_baselines.py --limit 10  # quick Ollama run
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
DEFAULT_OUTPUT = DEFAULT_CHECKPOINT / "compare_results.json"

from config import LABELS  # noqa: E402
from eval_classifier import compute_metrics, load_val_rows  # noqa: E402
from inference import build_prompt, load_finetuned, normalize_prediction, predict_batch  # noqa: E402
from shared.ollama_client import chat, chat_model, is_available  # noqa: E402

OLLAMA_SYSTEM = (
    "You classify rights clearance snippets into exactly one tier. "
    "Reply with only one word: cleared, pending, denied, or unknown."
)


def predict_ollama_zero_shot(snippet: str, model: str | None = None) -> tuple[str, float]:
    prompt = (
        f"{build_prompt(snippet)}\n\n"
        "Answer with exactly one word: cleared, pending, denied, or unknown."
    )
    raw, elapsed = chat(prompt, system=OLLAMA_SYSTEM, model=model, num_predict=16)
    return normalize_prediction(raw), elapsed


def run_ollama_baseline(
    rows: list[dict],
    *,
    model: str | None = None,
) -> tuple[list[str], float]:
    predictions: list[str] = []
    total_latency = 0.0
    for row in rows:
        pred, elapsed = predict_ollama_zero_shot(row["text"], model=model)
        predictions.append(pred)
        total_latency += elapsed
    return predictions, total_latency


def compare(
    *,
    adapter_dir: Path,
    val_path: Path,
    output_path: Path,
    limit: int,
    skip_ollama: bool,
    ollama_model: str | None,
) -> dict:
    rows = load_val_rows(val_path)
    if limit > 0:
        rows = rows[:limit]
    if not rows:
        raise SystemExit("No evaluation rows")

    y_true = [row["label"] for row in rows]

    model, tokenizer, device = load_finetuned(adapter_dir)
    finetuned_pred, finetuned_latency = predict_batch(
        [row["text"] for row in rows], model, tokenizer, device
    )
    finetuned_metrics = compute_metrics(y_true, finetuned_pred)

    result: dict = {
        "val_split": str(val_path),
        "n_samples": len(rows),
        "finetuned": {
            "adapter_dir": str(adapter_dir),
            "device": device,
            "model_id": "google/flan-t5-base + LoRA",
            "latency_ms_total": round(finetuned_latency * 1000, 2),
            "latency_ms_per_snippet": round((finetuned_latency / len(rows)) * 1000, 2),
            "cost_usd_est": 0.0,
            **finetuned_metrics,
        },
    }

    if skip_ollama:
        result["ollama_zero_shot"] = {"skipped": True}
    else:
        if not is_available():
            raise SystemExit(
                "Ollama not reachable. Run: ollama serve && ollama pull llama3.2\n"
                "Or pass --skip-ollama to compare finetuned metrics only."
            )
        model_name = ollama_model or chat_model()
        ollama_pred, ollama_latency = run_ollama_baseline(rows, model=model_name)
        ollama_metrics = compute_metrics(y_true, ollama_pred)
        result["ollama_zero_shot"] = {
            "model": model_name,
            "latency_ms_total": round(ollama_latency * 1000, 2),
            "latency_ms_per_snippet": round((ollama_latency / len(rows)) * 1000, 2),
            "cost_usd_est": 0.0,
            **ollama_metrics,
        }
        ft = result["finetuned"]
        ol = result["ollama_zero_shot"]
        result["summary"] = {
            "accuracy_delta_finetuned_minus_ollama": round(
                ft["accuracy"] - ol["accuracy"], 4
            ),
            "macro_f1_delta_finetuned_minus_ollama": round(
                ft["macro_f1"] - ol["macro_f1"], 4
            ),
            "latency_ratio_ollama_over_finetuned": round(
                ol["latency_ms_per_snippet"] / max(ft["latency_ms_per_snippet"], 0.01),
                2,
            ),
            "finetuned_wins_accuracy": ft["accuracy"] >= ol["accuracy"],
            "finetuned_wins_latency": ft["latency_ms_per_snippet"] <= ol["latency_ms_per_snippet"],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    ft = result["finetuned"]
    print(f"Samples: {len(rows)}")
    print("\nFine-tuned LoRA (flan-t5-base):")
    print(f"  Accuracy:  {ft['accuracy']:.3f}  Macro F1: {ft['macro_f1']:.3f}")
    print(f"  Latency:   {ft['latency_ms_per_snippet']:.1f} ms/snippet")

    if "ollama_zero_shot" in result and not result["ollama_zero_shot"].get("skipped"):
        ol = result["ollama_zero_shot"]
        summary = result["summary"]
        print(f"\nZero-shot Ollama ({ol['model']}):")
        print(f"  Accuracy:  {ol['accuracy']:.3f}  Macro F1: {ol['macro_f1']:.3f}")
        print(f"  Latency:   {ol['latency_ms_per_snippet']:.1f} ms/snippet")
        print("\nComparison:")
        print(f"  Δ accuracy (FT − Ollama): {summary['accuracy_delta_finetuned_minus_ollama']:+.3f}")
        print(f"  Δ macro F1   (FT − Ollama): {summary['macro_f1_delta_finetuned_minus_ollama']:+.3f}")
        print(f"  Ollama latency / FT latency: {summary['latency_ratio_ollama_over_finetuned']:.1f}x")

    print(f"\nWrote → {output_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LoRA vs zero-shot Ollama")
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=DEFAULT_CHECKPOINT / "adapter",
    )
    parser.add_argument("--val-split", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only first N val rows (useful for quick Ollama runs)",
    )
    parser.add_argument("--skip-ollama", action="store_true")
    parser.add_argument("--ollama-model", type=str, default=None)
    args = parser.parse_args()

    if not args.adapter_dir.exists():
        raise SystemExit(f"Adapter not found: {args.adapter_dir}")
    if not args.val_split.exists():
        raise SystemExit(f"Val split not found: {args.val_split}")

    compare(
        adapter_dir=args.adapter_dir,
        val_path=args.val_split,
        output_path=args.output,
        limit=args.limit,
        skip_ollama=args.skip_ollama,
        ollama_model=args.ollama_model,
    )


if __name__ == "__main__":
    main()
