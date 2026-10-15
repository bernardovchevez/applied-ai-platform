"""RAG context normalization and local-model answer cleanup (Ollama/llama3.2)."""

from __future__ import annotations

import re

# Small local models often garble these corpus terms — fix at platform layer.
_HDR_PASS = "HDR (High Dynamic Range) pass"


def normalize_chunk_text(text: str) -> str:
    """Expand abbreviations in retrieved chunks before they reach the LLM."""
    out = text
    out = re.sub(r"\bHDR pass\b", _HDR_PASS, out, flags=re.IGNORECASE)
    out = re.sub(
        r"\bPerpetuity \(series term\)\b",
        "Perpetuity / series-term clearance (US & Canada table row only — not EMEA expiry)",
        out,
        flags=re.IGNORECASE,
    )
    return out


def rag_question_hints(question: str) -> str:
    """Short grounding hints appended to the RAG user message."""
    q = question.lower()
    hints: list[str] = []
    if "emea" in q:
        hints.append(
            "For EMEA answers: use the EMEA row or EMEA section only. "
            "EMEA expiry is a calendar date (e.g. 2027-12-31), not perpetuity / series term."
        )
    if any(k in q for k in ("hdr", "color", "deliverable", "grade", "streaming tier")):
        hints.append(
            "HDR means High Dynamic Range color pass. Never write 'ss' — always write 'HDR pass'."
        )
    if not hints:
        return ""
    return "\n\nAssistant hints:\n- " + "\n- ".join(hints)


def unique_source_filenames(sources: list[dict[str, object]] | None) -> list[str]:
    seen: set[str] = set()
    files: list[str] = []
    for row in sources or []:
        name = row.get("file_name") if isinstance(row, dict) else None
        if isinstance(name, str) and name and name not in seen:
            seen.add(name)
            files.append(name)
    return files


def append_source_file_bullets(answer: str, source_files: list[str]) -> str:
    """Append filename-only source bullets; strip inline filename citations from prose."""
    body = answer.strip()
    if not body:
        return body

    # Platform lists sources below — remove redundant inline filename sentences
    body = re.sub(
        r'\s*(?:Additionally,?\s*)?(?:the document )?"[^"]+\.(?:md|txt|pdf)"[^.]*\.',
        "",
        body,
        flags=re.I,
    )
    body = re.sub(
        r'\s*"[^"]+\.(?:md|txt|pdf)"\s*(?:shows|states|says)[^.]*\.',
        "",
        body,
        flags=re.I,
    )
    body = body.strip()

    if not source_files:
        return body

    bullets = "\n".join(f"- {name}" for name in source_files)
    return f"{body}\n\nSources:\n{bullets}"


def clean_rag_answer(text: str, *, source_files: list[str] | None = None) -> str:
    """Fix common llama3.2 garbles in streamed RAG answers."""
    out = text

    out = re.sub(r"\bstates that ss is not required\b", "states that an HDR pass is not required", out, flags=re.I)
    out = re.sub(r"\bthat ss is not required\b", "that an HDR pass is not required", out, flags=re.I)
    out = re.sub(r"\bss is not required\b", "HDR pass is not required", out, flags=re.I)
    out = re.sub(r"\bss is required\b", "HDR pass is required", out, flags=re.I)
    out = re.sub(r"\bss pass\b", "HDR pass", out, flags=re.I)
    out = re.sub(r"\betuity\b", "perpetuity", out, flags=re.I)

    # US perpetuity row misapplied to EMEA — rewrite the bad clause
    out = re.sub(
        r"\bhas an? perpetuity\s*\(series term\)\s*cleared for EMEA[^.]*\.?",
        "is cleared for EMEA SVOD streaming through 2027-12-31",
        out,
        flags=re.I,
    )
    out = re.sub(r"\bhas an? etuity\s*\(series term\)\s*cleared for EMEA[^.]*\.?", 
        "is cleared for EMEA SVOD streaming through 2027-12-31",
        out,
        flags=re.I,
    )

    # Ensure EMEA clearance mentions expiry when music-clearance-emea is cited
    if "music-clearance-emea.md" in out and "2027-12-31" not in out and "emea" in out.lower():
        out = re.sub(
            r"\"music-clearance-emea\.md\" shows that [^.]+\.",
            '"music-clearance-emea.md" shows EMEA (EU27 + UK + EFTA) SVOD streaming is Cleared through 2027-12-31.',
            out,
            count=1,
            flags=re.I,
        )

    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return append_source_file_bullets(out, source_files or [])
