#!/usr/bin/env python3
"""Generate labeled rights-tier snippets for LoRA classifier.

Labels: cleared | pending | denied | unknown

Usage:
  python platform/finetune/generate_dataset.py
  python platform/finetune/generate_dataset.py --target 200 --stats
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
LAB_ROOT = FINETUNE_DIR.parent.parent
sys.path.insert(0, str(LAB_ROOT))
sys.path.insert(0, str(FINETUNE_DIR))

from stores.clearance_rows import normalize_row_status
from stores.corpus import iter_corpus_chunks, load_documents, load_sharepoint_meta

from config import LABELS

DATASET_DIR = FINETUNE_DIR / "dataset"
DEFAULT_OUTPUT = DATASET_DIR / "rights_tier.jsonl"
DEFAULT_AUDIT = DATASET_DIR / "audit_sample.json"

# Markdown table row: | col | col | col | col |
TABLE_ROW_RE = re.compile(
    r"^\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*([^|]*?)\s*\|\s*$"
)
STATUS_BULLET_RE = re.compile(
    r"^\s*[-*]?\s*\**Status:?\**\s*(.+)$",
    re.IGNORECASE | re.MULTILINE,
)
HEADER_SKIP = re.compile(r"^-+$|^:?-+:?$")

TEMPLATE_VARIANTS = (
    "{territory} | {platform} | {status_phrase} | {through}",
    "Harbor Line — {territory} {platform} clearance: {status_phrase}",
    "{series} theme song {territory} streaming ({platform}): {status_phrase}",
    "Rights memo excerpt: {territory} / {platform} / expires {through} — {status_phrase}",
    "Clearance row: {territory} | {platform} | {status_phrase}",
)


def to_classifier_label(raw_status: str) -> str:
    canonical = normalize_row_status(raw_status)
    if canonical == "excluded":
        return "denied"
    if canonical in LABELS:
        return canonical
    return "unknown"


def status_phrase(label: str) -> str:
    return {
        "cleared": "cleared",
        "pending": "pending review",
        "denied": "denied / excluded",
        "unknown": "status not specified",
    }.get(label, "unknown")


def _example_id(text: str, label: str, source: str) -> str:
    digest = hashlib.sha256(f"{text}|{label}|{source}".encode()).hexdigest()
    return digest[:16]


def _add(
    rows: list[dict],
    seen: set[str],
    *,
    text: str,
    label: str,
    source: str,
    series: str | None,
    territory: str | None,
    origin: str,
) -> None:
    text = " ".join(text.split())
    if len(text) < 12 or len(text) > 400:
        return
    label = to_classifier_label(label) if label not in LABELS else label
    if label not in LABELS:
        label = "unknown"
    key = hashlib.sha256(text.lower().encode()).hexdigest()
    if key in seen:
        return
    seen.add(key)
    rows.append(
        {
            "id": _example_id(text, label, source),
            "text": text,
            "label": label,
            "source": source,
            "series": series or "Harbor Line",
            "territory": (territory or "").lower() or None,
            "origin": origin,
        }
    )


def examples_from_warehouse_meta(seen: set[str]) -> list[dict]:
    rows: list[dict] = []
    meta_by_file = load_sharepoint_meta()
    for file_name, meta in meta_by_file.items():
        if meta.get("doc_type") != "rights_clearance":
            continue
        series = meta.get("series") or "Harbor Line"
        for item in meta.get("clearance_territories") or []:
            if not isinstance(item, dict):
                continue
            territory = str(item.get("territory") or "").lower()
            platform = str(item.get("platform") or "svod")
            status = str(item.get("status") or "")
            through = str(item.get("through_date") or "—")
            label = to_classifier_label(status)
            snippet = f"{territory} | {platform} | {status_phrase(label)} | {through}"
            _add(
                rows,
                seen,
                text=snippet,
                label=label,
                source=file_name,
                series=series,
                territory=territory,
                origin="warehouse",
            )
        territory_status = meta.get("territory_status") or {}
        if isinstance(territory_status, dict):
            for territory, status in territory_status.items():
                label = to_classifier_label(str(status))
                snippet = f"{territory.lower()} | all | {status_phrase(label)} | —"
                _add(
                    rows,
                    seen,
                    text=snippet,
                    label=label,
                    source=file_name,
                    series=series,
                    territory=str(territory).lower(),
                    origin="warehouse",
                )
    return rows


def examples_from_markdown_tables(seen: set[str]) -> list[dict]:
    rows: list[dict] = []
    for doc in load_documents():
        meta = doc["metadata"]
        if meta.get("doc_type") != "rights_clearance":
            continue
        source = meta.get("source") or ""
        series = meta.get("series") or "Harbor Line"
        for line in doc["text"].splitlines():
            match = TABLE_ROW_RE.match(line.strip())
            if not match:
                continue
            cols = [c.strip().strip("*") for c in match.groups()]
            if any(HEADER_SKIP.match(c.replace(" ", "")) for c in cols):
                continue
            if cols[0].lower() in {"territory", "territory group", "field", "episode"}:
                continue
            territory = cols[0].split("(")[0].strip().lower()
            platform = cols[1].lower() if len(cols) > 1 else "svod"
            # Product-placement tables use Episode | Scene | Duration | Status.
            if re.match(r"^s\d+e\d+$", territory):
                status_raw = cols[3] if len(cols) > 3 else ""
                through = cols[2] if len(cols) > 2 else "—"
            else:
                status_raw = cols[2] if len(cols) > 2 else ""
                through = cols[3] if len(cols) > 3 else "—"
            if not status_raw or status_raw.lower() in {"status", "platform"}:
                continue
            label = to_classifier_label(status_raw)
            snippet = f"{territory} | {platform} | {status_phrase(label)} | {through or '—'}"
            _add(
                rows,
                seen,
                text=snippet,
                label=label,
                source=source,
                series=series,
                territory=territory.split()[0] if territory else None,
                origin="table",
            )
    return rows


def examples_from_status_bullets(seen: set[str]) -> list[dict]:
    rows: list[dict] = []
    for doc in load_documents():
        meta = doc["metadata"]
        if meta.get("doc_type") != "rights_clearance":
            continue
        source = meta.get("source") or ""
        series = meta.get("series") or "Harbor Line"
        for match in STATUS_BULLET_RE.finditer(doc["text"]):
            phrase = match.group(1).strip().strip("*")
            label = to_classifier_label(phrase)
            territory = None
            if meta.get("clearance_territories"):
                territory = str(meta["clearance_territories"][0].get("territory", "")).lower()
            snippet = phrase[:280]
            _add(
                rows,
                seen,
                text=snippet,
                label=label,
                source=source,
                series=series,
                territory=territory,
                origin="bullet",
            )
    return rows


def examples_from_chunks(seen: set[str]) -> list[dict]:
    rows: list[dict] = []
    for _chunk_id, text, meta in iter_corpus_chunks():
        doc_type = meta.get("doc_type")
        source = meta.get("source") or ""
        series = meta.get("series") or "Harbor Line"
        lowered = text.lower()
        if doc_type == "rights_clearance":
            labels: set[str] = set()
            if "excluded" in lowered:
                labels.add("denied")
            if "not cleared" in lowered or re.search(r"\bpending\b", lowered):
                labels.add("pending")
            if "cleared" in lowered and "not cleared" not in lowered:
                labels.add("cleared")
            if len(labels) > 1:
                label = "unknown"
            elif len(labels) == 1:
                label = next(iter(labels))
            elif meta.get("rights_tier"):
                label = to_classifier_label(str(meta["rights_tier"]))
            else:
                label = "unknown"
            territory = None
            for tag in meta.get("tags") or []:
                if tag.lower() in {"emea", "latam", "apac", "japan", "korea", "china"}:
                    territory = tag.lower()
                    break
            _add(
                rows,
                seen,
                text=text[:320],
                label=label,
                source=source,
                series=series,
                territory=territory,
                origin="chunk",
            )
        elif doc_type in {"production_brief", "press_release", "press_kit", "location_report"}:
            if any(w in lowered for w in ("cleared", "pending", "excluded", "not cleared")):
                continue
            _add(
                rows,
                seen,
                text=text[:280],
                label="unknown",
                source=source,
                series=series,
                territory=None,
                origin="negative",
            )
    return rows


def augment_templates(seeds: list[dict], seen: set[str], rng: random.Random) -> list[dict]:
    rows: list[dict] = []
    for seed in seeds:
        if seed["origin"] == "negative":
            continue
        territory = seed.get("territory") or rng.choice(["emea", "latam", "apac", "japan", "us"])
        platform = rng.choice(["svod", "linear+svod", "all", "avod"])
        through = rng.choice(["2027-12-31", "2026-08-15", "perpetuity", "—"])
        label = seed["label"]
        for tmpl in TEMPLATE_VARIANTS:
            text = tmpl.format(
                territory=territory,
                platform=platform,
                status_phrase=status_phrase(label),
                through=through,
                series=seed.get("series") or "Harbor Line",
            )
            _add(
                rows,
                seen,
                text=text,
                label=label,
                source=seed["source"],
                series=seed.get("series"),
                territory=territory,
                origin="template",
            )
    return rows


def build_dataset(target: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    seen: set[str] = set()
    rows: list[dict] = []

    for collector in (
        examples_from_warehouse_meta,
        examples_from_markdown_tables,
        examples_from_status_bullets,
        examples_from_chunks,
    ):
        rows.extend(collector(seen))

    seeds = list(rows)
    while len(rows) < target and seeds:
        before = len(rows)
        rows.extend(augment_templates(seeds, seen, rng))
        if len(rows) == before:
            break

    rng.shuffle(rows)
    return rows[:target] if len(rows) > target else rows


def label_counts(rows: list[dict]) -> dict[str, int]:
    counts = {label: 0 for label in LABELS}
    for row in rows:
        counts[row["label"]] = counts.get(row["label"], 0) + 1
    return counts


def select_audit_sample(rows: list[dict], n: int = 20) -> list[dict]:
    """Prefer non-template rows for hand audit."""
    priority = {"warehouse": 0, "table": 1, "bullet": 2, "chunk": 3, "negative": 4, "template": 5}
    per_label = {label: [] for label in LABELS}
    for row in rows:
        per_label[row["label"]].append(row)
    for label in per_label:
        per_label[label].sort(key=lambda r: (priority.get(r["origin"], 9), r["text"]))

    sample: list[dict] = []
    quota = max(1, n // len(LABELS))
    for label in LABELS:
        sample.extend(per_label[label][:quota])
    remaining = n - len(sample)
    pool = [r for r in rows if r not in sample]
    pool.sort(key=lambda r: priority.get(r["origin"], 9))
    sample.extend(pool[:remaining])
    return sample[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate rights tier classification dataset")
    parser.add_argument("--target", type=int, default=200, help="Target example count")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSONL path",
    )
    parser.add_argument("--stats", action="store_true", help="Print label/origin breakdown")
    parser.add_argument(
        "--audit",
        type=Path,
        default=DEFAULT_AUDIT,
        help="Write hand-audit sample JSON (20 rows, non-template preferred)",
    )
    args = parser.parse_args()

    rows = build_dataset(target=args.target, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = select_audit_sample(rows)
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    counts = label_counts(rows)
    print(f"Wrote {len(rows)} examples → {args.output}")
    print(f"Wrote {len(audit)} audit rows → {args.audit}")
    print(f"Labels: {counts}")
    if args.stats:
        origins: dict[str, int] = {}
        for row in rows:
            origins[row["origin"]] = origins.get(row["origin"], 0) + 1
        print(f"Origins: {origins}")


if __name__ == "__main__":
    main()
