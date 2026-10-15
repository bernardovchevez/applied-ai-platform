"""Load LoRA adapter and predict rights tier labels."""

from __future__ import annotations

import re
import time
from pathlib import Path

from config import LABELS, MAX_INPUT_LENGTH, MAX_LABEL_LENGTH, MODEL_ID, TASK_PREFIX
from stores.clearance_rows import normalize_row_status


def build_prompt(snippet: str) -> str:
    return f"{TASK_PREFIX} {snippet}"


def normalize_prediction(text: str) -> str:
    lowered = re.sub(r"[^\w\s/]", " ", text.strip().lower())
    lowered = " ".join(lowered.split())
    if lowered in LABELS:
        return lowered
    for label in LABELS:
        if re.search(rf"\b{re.escape(label)}\b", lowered):
            return label
    canonical = normalize_row_status(lowered)
    if canonical == "excluded":
        return "denied"
    if canonical in LABELS:
        return canonical
    return "unknown"


def pick_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_finetuned(adapter_dir: Path, device: str | None = None):
    from peft import PeftModel
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = device or pick_device()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    model.to(device)
    return model, tokenizer, device


def predict_tier(
    snippet: str,
    model,
    tokenizer,
    device: str,
) -> tuple[str, float]:
    import torch

    prompt = build_prompt(snippet)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=MAX_LABEL_LENGTH)
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return normalize_prediction(text), elapsed


def predict_batch(
    snippets: list[str],
    model,
    tokenizer,
    device: str,
) -> tuple[list[str], float]:
    predictions: list[str] = []
    start = time.perf_counter()
    for snippet in snippets:
        pred, _ = predict_tier(snippet, model, tokenizer, device)
        predictions.append(pred)
    total = time.perf_counter() - start
    return predictions, total
