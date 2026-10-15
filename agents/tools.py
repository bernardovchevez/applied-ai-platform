"""LangChain tools over RAG + pgvector for media workflow agents."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

from langchain_core.tools import tool

LAB_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(LAB_ROOT))

from shared.llm import chat_completion
from stores.clearance import CONFIDENCE_THRESHOLD, resolve_clearance_status
from stores.pgvector import PgVectorStore

# Canonical territory codes for tag/filename matching
KNOWN_TERRITORIES = ("emea", "latam", "apac", "china", "us")

TERRITORY_ALIASES = {
    "europe": "emea",
    "eu27": "emea",
    "uk": "emea",
    "efta": "emea",
    "latin america": "latam",
    "asia pacific": "apac",
}

# Territories without a dedicated memo file: match territory name in chunk body
TEXT_BODY_TERRITORY_MATCH = frozenset({"antarctica"})

# APAC subregions map to APAC clearance memos (Japan/Korea in apac-music-clearance-update.md)
SUBREGION_TO_TERRITORY = {
    "japan": "apac",
    "korea": "apac",
}


def normalize_territory(territory: str) -> str:
    """Map agent phrases like 'EMEA streaming launch' → 'emea'."""
    raw = territory.lower().strip()
    if raw in TERRITORY_ALIASES:
        return TERRITORY_ALIASES[raw]
    if raw in SUBREGION_TO_TERRITORY:
        return SUBREGION_TO_TERRITORY[raw]
    if raw in KNOWN_TERRITORIES:
        return raw

    for sub, canonical in SUBREGION_TO_TERRITORY.items():
        if re.search(rf"\b{re.escape(sub)}\b", raw):
            return canonical

    # Prefer explicit territory codes (emea before accidental 'la' in 'launch')
    for code in sorted(KNOWN_TERRITORIES, key=len, reverse=True):
        if re.search(rf"\b{re.escape(code)}\b", raw):
            return code

    for alias, canonical in sorted(TERRITORY_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in raw:
            return canonical

    first = raw.split()[0] if raw else raw
    if first in SUBREGION_TO_TERRITORY:
        return SUBREGION_TO_TERRITORY[first]
    return TERRITORY_ALIASES.get(first, first if first in KNOWN_TERRITORIES else first)


def _territory_match_keys(territory_key: str) -> set[str]:
    keys = {territory_key}
    for alias, canonical in TERRITORY_ALIASES.items():
        if canonical == territory_key:
            keys.add(alias)
    return keys


def _load_query_module(name: str, filename: str):
    query_path = LAB_ROOT / "rag" / filename
    spec = importlib.util.spec_from_file_location(name, query_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def lookup_rights_impl_for_guard(series: str, territory: str) -> dict:
    """Expose clearance lookup for agent guards (no circular import with @tool)."""
    return _lookup_rights_impl(series, territory)


def search_docs_impl_for_guard(question: str) -> dict:
    """Expose RAG lookup for agent grounding guard (no circular import with @tool)."""
    return _search_docs_impl(question)


def _search_docs_impl(question: str, top_k: int = 4) -> dict:
    """RAG search: prefers pgvector, falls back to Chroma."""
    try:
        mod = _load_query_module("rag_pgvector", "query_pgvector.py")
        return mod.answer(question, top_k=top_k)
    except Exception:
        mod = _load_query_module("rag_chroma", "query.py")
        return mod.answer(question, top_k=top_k)


def _territory_status_for_hit(hit, territory_key: str) -> str | None:
    ts = hit.metadata.get("territory_status") or {}
    if not isinstance(ts, dict):
        return None
    for key, value in ts.items():
        if str(key).lower() == territory_key:
            return str(value).lower()
    return None


def _territories_for_sql(territory_key: str, raw_territory: str) -> list[str]:
    """Territory codes to match in clearance_territories warehouse table."""
    keys = {territory_key}
    raw = raw_territory.lower()
    for sub in ("japan", "korea", "china"):
        if re.search(rf"\b{re.escape(sub)}\b", raw):
            keys.add(sub)
    if territory_key == "apac":
        keys.update({"japan", "korea", "apac"})
    return sorted(keys)


def _lookup_from_sql_rows(
    series: str,
    territory: str,
    territory_key: str,
    sql_rows: list[dict],
) -> dict:
    territory_statuses = [str(r["status"]).lower() for r in sql_rows if r.get("status")]
    tiers = {str(r.get("rights_tier") or "unknown").lower() for r in sql_rows}
    clearance_records = []
    for row in sql_rows:
        clearance_records.append(
            {
                "source": row.get("source"),
                "rights_tier": row.get("rights_tier"),
                "territory_status": row.get("status"),
                "series": row.get("series"),
                "sharepoint_path": row.get("sharepoint_path"),
                "snippet": (
                    f"{row.get('territory')} | {row.get('platform') or 'n/a'} | "
                    f"{row.get('status')} | {row.get('through_date') or 'n/a'}"
                )[:280],
                "distance": 0.0,
            }
        )
    status, confidence = resolve_clearance_status(
        territory_statuses,
        tiers,
        has_territory_match=True,
    )
    return {
        "series": series,
        "territory": territory,
        "territory_normalized": territory_key,
        "status": status,
        "confidence": confidence,
        "needs_escalation": confidence < CONFIDENCE_THRESHOLD,
        "clearance_records": clearance_records,
        "lookup_method": "sql",
    }


def _lookup_rights_vector(series: str, territory: str, territory_key: str) -> dict:
    """Fallback: vector retrieval + chunk metadata when SQL rows are absent."""
    store = PgVectorStore()
    match_keys = _territory_match_keys(territory_key)
    query = f"{series} music and content clearance status {territory_key} streaming"
    hits = store.search(query, top_k=8, doc_type="rights_clearance")

    series_key = series.lower()

    def matches_territory(hit) -> bool:
        source = (hit.metadata.get("source") or "").lower()
        tags = [str(t).lower() for t in (hit.metadata.get("tags") or [])]
        text = hit.text.lower()
        ts = hit.metadata.get("territory_status") or {}
        if isinstance(ts, dict) and any(str(k).lower() == territory_key for k in ts):
            return True
        if any(key in tags for key in match_keys):
            return True
        if any(key in source for key in match_keys):
            return True
        if territory_key in TEXT_BODY_TERRITORY_MATCH and re.search(
            rf"\b{re.escape(territory_key)}\b", text
        ):
            return True
        return False

    def matches_series(hit) -> bool:
        series_name = str(hit.metadata.get("series") or "").lower()
        if not series_name:
            return True
        return series_key in series_name or series_name in series_key

    by_territory = [h for h in hits if matches_territory(h) and matches_series(h)]
    if by_territory:
        records = by_territory
    else:
        records = [h for h in hits if matches_series(h)][:4] or hits[:4]

    clearance_rows = []
    tiers: set[str] = set()
    territory_row_statuses: list[str] = []
    for hit in records:
        row_status = _territory_status_for_hit(hit, territory_key)
        if row_status:
            territory_row_statuses.append(row_status)
        tier = hit.metadata.get("rights_tier") or "unknown"
        if tier:
            tiers.add(tier)
        clearance_rows.append(
            {
                "source": hit.metadata.get("source"),
                "rights_tier": tier,
                "territory_status": row_status,
                "series": hit.metadata.get("series"),
                "sharepoint_path": hit.metadata.get("sharepoint_path"),
                "snippet": hit.text[:280],
                "distance": round(hit.distance, 4),
            }
        )

    status, confidence = resolve_clearance_status(
        territory_row_statuses,
        tiers,
        has_territory_match=bool(by_territory),
    )

    return {
        "series": series,
        "territory": territory,
        "territory_normalized": territory_key,
        "status": status,
        "confidence": confidence,
        "needs_escalation": confidence < CONFIDENCE_THRESHOLD,
        "clearance_records": clearance_rows,
        "lookup_method": "vector",
    }


def _lookup_rights_impl(series: str, territory: str) -> dict:
    """Structured clearance lookup: SQL warehouse rows first, vector fallback."""
    store = PgVectorStore()
    territory_key = normalize_territory(territory)
    sql_territories = _territories_for_sql(territory_key, territory)
    sql_rows = store.lookup_clearance_territories(series, sql_territories)
    if sql_rows:
        return _lookup_from_sql_rows(series, territory, territory_key, sql_rows)
    return _lookup_rights_vector(series, territory, territory_key)


def draft_brief_impl_for_guard(topic: str, context: str) -> str:
    """Expose brief drafting for agent auto-draft (no circular import with @tool)."""
    return _draft_brief_impl(topic, context)


def _draft_brief_impl(topic: str, context: str) -> str:
    system = (
        "You are a production coordinator at an enterprise media company. "
        "Write a concise production brief with sections: Summary, Clearances, "
        "Open items, Recommended actions. Cite source filenames inline."
    )
    user = f"Topic: {topic}\n\nResearch context:\n{context}"
    return chat_completion(user, system=system, max_tokens=800)


def escalate_legal_impl_for_guard(reason: str, confidence: float) -> dict:
    """Expose escalation for agent auto-escalate (no circular import with @tool)."""
    return _escalate_legal_impl(reason, confidence)


def _escalate_legal_impl(reason: str, confidence: float) -> dict:
    return {
        "action": "escalate",
        "team": "legal",
        "reason": reason,
        "confidence": confidence,
        "threshold": CONFIDENCE_THRESHOLD,
        "message": "Route to Music Legal for manual review before distribution.",
    }


IP_REVIEW_PATTERN = re.compile(
    r"\b(sponsor|sponsorship|placement|product placement|brand|trademark|"
    r"intellectual property|\bip\b|logo|ad sales)\b",
    re.I,
)


def search_web_stub_impl(query: str) -> dict:
    """Web search stub: disabled in corpus-only mode."""
    return {
        "status": "disabled",
        "query": query,
        "answer": (
            "External web search is not enabled in this environment. "
            "Use internal document search (search_docs) over the SharePoint corpus."
        ),
        "sources": [],
    }


def _ip_hit_is_relevant(hit) -> bool:
    doc_type = (hit.metadata.get("doc_type") or "").lower()
    tags = [str(t).lower() for t in (hit.metadata.get("tags") or [])]
    source = (hit.metadata.get("source") or "").lower()
    if doc_type in {"sponsorship_deck", "production_brief", "rights_clearance"}:
        return True
    if any(k in tags for k in ("sponsorship", "placement", "brand")):
        return True
    if "product-placement" in source or "sponsorship" in source:
        return True
    return False


def check_ip_flags_impl(series: str, user_message: str) -> dict:
    """Rule-based IP / sponsorship flag scan over pgvector metadata."""
    store = PgVectorStore()
    query = f"{series} product placement sponsorship brand IP clearance restrictions"
    hits = store.search(query, top_k=6)

    flags: list[dict] = []
    for hit in hits:
        if not _ip_hit_is_relevant(hit):
            continue
        text = hit.text.lower()
        severity = "info"
        if any(w in text for w in ("do not", "not cleared", "hold", "restricted", "requires")):
            severity = "review"
        flags.append(
            {
                "source": hit.metadata.get("source"),
                "doc_type": hit.metadata.get("doc_type"),
                "severity": severity,
                "snippet": hit.text[:220],
            }
        )

    # Deduplicate by source
    seen: set[str] = set()
    unique_flags: list[dict] = []
    for row in flags:
        src = row.get("source") or ""
        if src in seen:
            continue
        seen.add(src)
        unique_flags.append(row)

    needs_review = any(f.get("severity") == "review" for f in unique_flags)
    return {
        "series": series,
        "flags": unique_flags[:4],
        "needs_review": needs_review,
        "scanned": bool(unique_flags),
    }


def should_scan_ip_flags(user_message: str) -> bool:
    if IP_REVIEW_PATTERN.search(user_message):
        return True
    return bool(
        re.search(r"\b(summary|prepare|brief|launch)\b", user_message, re.I)
    )


def compliance_llm_review_impl(
    user_message: str,
    *,
    lookup: dict | None,
    search: dict | None,
    ip_flags: dict | None,
) -> str | None:
    """Optional LLM compliance note when rights or IP flags need attention."""
    lookup = lookup or {}
    ip_flags = ip_flags or {}
    needs = (
        lookup.get("needs_escalation")
        or ip_flags.get("needs_review")
        or lookup.get("status") in ("pending", "unknown", "excluded")
    )
    if not needs:
        return None

    flag_lines = [
        f"- {f.get('source')} [{f.get('severity')}]: {(f.get('snippet') or '')[:160]}"
        for f in ip_flags.get("flags", [])[:3]
    ]
    context = (
        f"User request: {user_message}\n"
        f"Clearance: {lookup.get('status')} (confidence {lookup.get('confidence')})\n"
        f"IP flags: {len(ip_flags.get('flags', []))} hit(s), needs_review={ip_flags.get('needs_review')}\n"
    )
    if flag_lines:
        context += "Flag sources:\n" + "\n".join(flag_lines) + "\n"
    if search and search.get("answer"):
        context += f"Research excerpt: {str(search.get('answer'))[:400]}\n"

    system = (
        "You are a compliance reviewer at an enterprise media company. "
        "Write 2-3 bullet points on legal/IP/sponsorship risks and recommended actions. "
        "Cite source filenames. Be concise; do not invent facts beyond the context."
    )
    return chat_completion(context, system=system, max_tokens=250)


SUMMARY_FLOW_PATTERN = re.compile(
    r"\b(summary|prepare|production brief|clearance summary|draft brief)\b",
    re.I,
)

CLEARANCE_QUESTION_PATTERN = re.compile(
    r"\b(cleared|clearance|rights|streaming|svod)\b",
    re.I,
)


def critic_llm_review_impl(
    user_message: str,
    draft: str,
    *,
    lookup: dict | None,
    search: dict | None,
    deterministic_issues: list[str] | None = None,
) -> str | None:
    """LLM critic note: alignment, citations, launch readiness."""
    text = (draft or "").strip()
    if not text:
        return None

    lookup = lookup or {}
    issues = deterministic_issues or []
    is_summary = bool(SUMMARY_FLOW_PATTERN.search(user_message))
    is_clearance = bool(CLEARANCE_QUESTION_PATTERN.search(user_message)) and bool(lookup)

    if not (is_summary or is_clearance or issues):
        return None

    source_lines = [
        row.get("source")
        for row in lookup.get("clearance_records", [])
        if row.get("source")
    ]
    if search and search.get("sources"):
        source_lines.extend(search.get("sources") or [])

    context_parts = [
        f"User request: {user_message}",
        f"Draft to review:\n{text[:2200]}",
    ]
    if lookup:
        context_parts.append(
            f"lookup_rights: status={lookup.get('status')} "
            f"confidence={lookup.get('confidence')} "
            f"territory={lookup.get('territory_normalized') or lookup.get('territory')}"
        )
    if search and search.get("answer"):
        context_parts.append(f"search_docs answer excerpt: {str(search.get('answer'))[:500]}")
    if source_lines:
        unique_sources = list(dict.fromkeys(source_lines))[:8]
        context_parts.append(f"Known sources: {', '.join(unique_sources)}")
    if issues:
        context_parts.append(
            "Deterministic checks already flagged:\n"
            + "\n".join(f"- {issue}" for issue in issues)
        )

    system = (
        "You are a senior production reviewer at an enterprise media company. "
        "Review the draft against lookup_rights and search_docs context. "
        "Start with one line: VERDICT: APPROVED or VERDICT: CONCERNS. "
        "Then 2-4 bullets on factual alignment, citations, and launch readiness. "
        "Prefix concern bullets with CONCERN:. Do not invent facts beyond the context. "
        "If deterministic issues were flagged, acknowledge them; do not contradict tool JSON."
    )
    return chat_completion("\n\n".join(context_parts), system=system, max_tokens=280)


def parse_critic_llm_concerns(notes: str | None) -> list[str]:
    """Extract CONCERN: bullets from critic LLM output."""
    if not notes:
        return []
    concerns: list[str] = []
    for line in notes.splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CONCERN:"):
            concerns.append(stripped.split(":", 1)[1].strip())
        elif stripped.startswith("- CONCERN:"):
            concerns.append(stripped[len("- CONCERN:") :].strip())
    return concerns[:4]

@tool
def search_docs(question: str) -> str:
    """Search media ops documents (rights, production, ad sales, legal)."""
    return json.dumps(_search_docs_impl(question), default=str)


@tool
def lookup_rights(series: str, territory: str) -> str:
    """Look up structured music/content clearance for a series in a territory (pgvector + SQL metadata)."""
    return json.dumps(_lookup_rights_impl(series, territory), default=str)


@tool
def draft_brief(topic: str, context: str) -> str:
    """Draft a production brief for the production team from gathered research context."""
    return _draft_brief_impl(topic, context)


@tool
def escalate_legal(reason: str, confidence: float) -> str:
    """Escalate to legal when clearance is pending, unclear, or confidence is below 0.7."""
    return json.dumps(_escalate_legal_impl(reason, confidence), default=str)


def get_tools() -> list:
    return [search_docs, lookup_rights, draft_brief, escalate_legal]
