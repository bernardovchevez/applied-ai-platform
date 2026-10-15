"""Shared workflow state, guards, and post-processing for agent_v1 + graph."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

from shared.config import get_env, require_env
from shared.llm import llm_provider
from shared.ollama_client import is_available as ollama_available
from tools import (
    CONFIDENCE_THRESHOLD,
    draft_brief_impl_for_guard,
    escalate_legal_impl_for_guard,
    lookup_rights_impl_for_guard,
    search_docs_impl_for_guard,
)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "guardrails" / "prompts"

REFUSAL_MARKERS = (
    "don't have",
    "do not have",
    "not in the provided",
    "not in the documents",
    "cannot answer",
    "no information",
    "outside the scope",
    "provided documents",
    "unknown based on",
    "does not mention",
    "do not mention",
    "not mentioned",
    "does not address",
    "does not contain",
    "context does not mention",
    "context does not contain",
    "cannot determine from the",
    "cannot find in the",
)

# search_docs pivoted to unrelated corpus content but admitted the question is missing
OUT_OF_CORPUS_SIGNALS = (
    "does not mention",
    "do not mention",
    "not mentioned in",
    "not mentioned",
    "does not address",
    "does not contain",
    "no information about",
    "context does not mention",
    "context does not contain",
    "cannot find in the provided",
    "cannot determine from the provided",
)

STANDARD_REFUSAL = (
    "I don't have enough information in the provided documents to answer this question."
)

# Block obvious general-knowledge leaks after search_docs said the topic is absent
_GENERAL_KNOWLEDGE_LEAK = re.compile(
    r"\b(ted sarandos|greg peters|reed hastings|is the (?:current )?ceo of)\b",
    re.I,
)

SUMMARY_FLOW_PATTERN = re.compile(
    r"\b(summary|prepare|production brief|clearance summary|draft brief)\b",
    re.I,
)

CLEARANCE_TERRITORY_PATTERN = re.compile(
    r"\b("
    r"emea|latam|apac|china|japan|korea|antarctica|australia|"
    r"latin america|asia pacific|europe|uk|us"
    r")\b",
    re.I,
)

CLEARANCE_QUESTION_PATTERN = re.compile(
    r"\b(cleared|clearance|rights|streaming|svod)\b",
    re.I,
)


@dataclass
class AgentResult:
    answer: str
    lookup_rights: dict | None = None
    escalation: dict | None = None
    search_docs: dict | None = None
    draft_brief: str | None = None
    grounding_override: bool = False
    clearance_grounding_override: bool = False
    auto_escalated: bool = False
    auto_drafted: bool = False

    @property
    def full_output(self) -> str:
        footer = format_assessment_footer(self)
        if footer and footer not in self.answer:
            return f"{self.answer.rstrip()}\n\n{footer}"
        return self.answer


def get_llm():
    provider = llm_provider()
    if provider == "ollama":
        if not ollama_available():
            raise RuntimeError(
                "Ollama not reachable. Run: ollama serve && ollama pull llama3.2"
            )
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=get_env("OLLAMA_CHAT_MODEL", "llama3.2"),
            temperature=0,
        )

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=get_env("CHAT_MODEL", "gpt-4o-mini"),
        api_key=require_env("OPENAI_API_KEY"),
        temperature=0,
    )


def extract_final_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            text = msg.content
            if isinstance(text, str):
                return text
            if isinstance(text, list):
                parts = [p.get("text", "") for p in text if isinstance(p, dict)]
                return "\n".join(p for p in parts if p)
    return ""


def _parse_tool_payload(content: str | list) -> dict | None:
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    if not isinstance(content, str) or not content.strip():
        return None
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None


def is_refusal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def search_indicates_out_of_scope(text: str) -> bool:
    """True when RAG admitted the question is not covered (even if it pivoted to other facts)."""
    lowered = (text or "").lower()
    return any(signal in lowered for signal in OUT_OF_CORPUS_SIGNALS)


def contains_general_knowledge_leak(text: str) -> bool:
    return bool(_GENERAL_KNOWLEDGE_LEAK.search(text or ""))


def standard_refusal_message(_user_message: str = "") -> str:
    return STANDARD_REFUSAL


def should_force_refusal(search_answer: str, final_answer: str) -> bool:
    if contains_general_knowledge_leak(final_answer):
        return True
    if is_refusal(final_answer):
        return False
    if is_refusal(search_answer) or search_indicates_out_of_scope(search_answer):
        return True
    return False


def is_summary_flow(user_message: str) -> bool:
    return bool(SUMMARY_FLOW_PATTERN.search(user_message))


def _tool_message_text(content: str | list) -> str:
    if isinstance(content, list):
        return "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return str(content or "")


def extract_draft_brief(messages: list) -> str | None:
    latest: str | None = None
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        if getattr(msg, "name", "") != "draft_brief":
            continue
        text = _tool_message_text(msg.content).strip()
        if text:
            latest = text
    return latest


def extract_search_docs(messages: list) -> dict | None:
    latest: dict | None = None
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        data = _parse_tool_payload(msg.content)
        if data and "answer" in data and "sources" in data:
            latest = data
    return latest


def extract_tool_summaries(messages: list) -> tuple[dict | None, dict | None, dict | None]:
    lookup: dict | None = None
    escalation: dict | None = None
    search: dict | None = None
    for msg in messages:
        if not isinstance(msg, ToolMessage):
            continue
        data = _parse_tool_payload(msg.content)
        if not data:
            continue
        if "status" in data and "confidence" in data and "clearance_records" in data:
            lookup = data
        elif "answer" in data and "sources" in data:
            search = data
        if data.get("action") == "escalate":
            escalation = data
    return lookup, search, escalation


def infer_territory_from_message(user_message: str) -> str | None:
    match = CLEARANCE_TERRITORY_PATTERN.search(user_message)
    return match.group(1) if match else None


def infer_series_from_message(user_message: str) -> str:
    if re.search(r"\bharbor line\b", user_message, re.I):
        return "Harbor Line"
    return "Harbor Line"


def is_clearance_question(user_message: str) -> bool:
    return bool(CLEARANCE_QUESTION_PATTERN.search(user_message)) and bool(
        infer_territory_from_message(user_message)
    )


def ensure_lookup_rights(user_message: str, result: AgentResult) -> AgentResult:
    if result.lookup_rights or not is_clearance_question(user_message):
        return result
    territory = infer_territory_from_message(user_message)
    if not territory:
        return result
    result.lookup_rights = lookup_rights_impl_for_guard(
        infer_series_from_message(user_message),
        territory,
    )
    return result


def claims_cleared(answer: str) -> bool:
    lowered = (answer or "").lower()
    if "not cleared" in lowered or "isn't cleared" in lowered:
        return False
    return bool(re.search(r"\bcleared\b", lowered))


def claims_pending(answer: str) -> bool:
    lowered = (answer or "").lower()
    return bool(re.search(r"\bpending\b", lowered))


def answer_contradicts_lookup(answer: str, lookup: dict) -> bool:
    status = lookup.get("status")
    if status == "cleared" and claims_pending(answer) and not claims_cleared(answer):
        return True
    if status in ("pending", "unknown", "excluded") and claims_cleared(answer):
        return True
    if status == "unknown" and claims_pending(answer):
        return True
    return False


def format_clearance_answer(lookup: dict) -> str:
    series = lookup.get("series", "Harbor Line")
    territory = lookup.get("territory") or lookup.get("territory_normalized") or "territory"
    status = lookup.get("status", "unknown")
    confidence = lookup.get("confidence")
    sources = list(
        dict.fromkeys(
            row.get("source")
            for row in lookup.get("clearance_records", [])
            if row.get("source")
        )
    )
    src = ", ".join(sources[:3]) if sources else "clearance memos"

    if status == "excluded":
        return (
            f"**{series}** is **not cleared** for **{territory}** — territory is **excluded** "
            f"per distribution agreement (confidence {confidence}). Sources: {src}."
        )
    if status == "cleared":
        return (
            f"**{series}** is **cleared** for **{territory}** streaming "
            f"(confidence {confidence}). Sources: {src}."
        )
    if status == "pending":
        return (
            f"**{series}** is **not cleared** / **pending** for **{territory}** streaming "
            f"(confidence {confidence}). Do not launch until legal confirms. Sources: {src}."
        )
    return (
        f"Clearance for **{series}** in **{territory}** is **unknown** — no territory-specific memo "
        f"(confidence {confidence}). Escalate to legal before distribution."
    )


def apply_clearance_grounding_guard(result: AgentResult) -> AgentResult:
    lookup = result.lookup_rights
    if not lookup or not result.answer:
        return result
    if not answer_contradicts_lookup(result.answer, lookup):
        return result
    result.answer = format_clearance_answer(lookup)
    result.clearance_grounding_override = True
    return result


def apply_grounding_guard(user_message: str, result: AgentResult, messages: list) -> AgentResult:
    search = result.search_docs
    if search is None:
        search = extract_search_docs(messages)
        result.search_docs = search

    if search is None and result.lookup_rights is None:
        search = search_docs_impl_for_guard(user_message)
        result.search_docs = search

    if search:
        search_answer = str(search.get("answer", "")).strip()
        final_answer = (result.answer or "").strip()
        if should_force_refusal(search_answer, final_answer):
            result.answer = standard_refusal_message(user_message)
            result.grounding_override = True
        elif is_refusal(search_answer) and not is_refusal(final_answer):
            result.answer = search_answer
            result.grounding_override = True

    return result


def _escalation_reason(lookup: dict) -> str:
    series = lookup.get("series", "series")
    territory = lookup.get("territory_normalized") or lookup.get("territory", "territory")
    status = lookup.get("status", "unknown")
    return (
        f"{series} clearance in {territory} is {status} "
        f"(confidence {lookup.get('confidence')}) — below threshold {CONFIDENCE_THRESHOLD}"
    )


def format_escalation_block(escalation: dict, *, auto: bool = False) -> str:
    label = "Legal escalation (auto-triggered)" if auto else "Legal escalation"
    return (
        f"---\n"
        f"**{label}**\n"
        f"- **Team:** {escalation.get('team', 'legal')}\n"
        f"- **Reason:** {escalation.get('reason')}\n"
        f"- **Confidence:** {escalation.get('confidence')} "
        f"(threshold {escalation.get('threshold', CONFIDENCE_THRESHOLD)})\n"
        f"- **Action:** {escalation.get('message')}"
    )


def build_brief_context(user_message: str, result: AgentResult) -> str:
    parts = [f"User request: {user_message}"]

    lookup = result.lookup_rights
    if lookup:
        parts.append(
            f"Clearance: {lookup.get('status')} "
            f"(confidence {lookup.get('confidence')}, "
            f"territory {lookup.get('territory_normalized') or lookup.get('territory')})"
        )
        for row in lookup.get("clearance_records", [])[:4]:
            parts.append(
                f"- {row.get('source')} [{row.get('rights_tier')}]: "
                f"{(row.get('snippet') or '')[:220]}"
            )

    search = result.search_docs
    if search:
        parts.append(f"RAG answer: {search.get('answer', '')[:1200]}")
        sources = search.get("sources") or []
        if sources:
            parts.append(f"RAG sources: {', '.join(sources[:6])}")

    if result.escalation:
        parts.append(f"Legal escalation: {result.escalation.get('reason')}")

    if len(parts) == 1 and result.answer:
        parts.append(f"Agent notes: {result.answer[:1500]}")

    return "\n\n".join(parts)


def apply_auto_draft_brief(user_message: str, result: AgentResult) -> AgentResult:
    if result.draft_brief or not is_summary_flow(user_message):
        return result
    if is_refusal(result.answer) and not result.lookup_rights:
        return result

    context = build_brief_context(user_message, result)
    brief = draft_brief_impl_for_guard(user_message, context)
    result.draft_brief = brief
    result.auto_drafted = True
    result.answer = brief
    return result


def apply_auto_escalation(result: AgentResult) -> AgentResult:
    if result.escalation or not result.lookup_rights:
        return result

    lookup = result.lookup_rights
    confidence = float(lookup.get("confidence", 1.0))
    needs = lookup.get("needs_escalation") or confidence < CONFIDENCE_THRESHOLD
    if not needs:
        return result

    reason = _escalation_reason(lookup)
    result.escalation = escalate_legal_impl_for_guard(reason, confidence)
    result.auto_escalated = True

    block = format_escalation_block(result.escalation, auto=True)
    if block not in result.answer:
        result.answer = f"{result.answer.rstrip()}\n\n{block}"

    return result


def format_assessment_footer(result: AgentResult) -> str:
    lookup = result.lookup_rights
    if not lookup:
        if result.grounding_override or (
            result.search_docs and is_refusal(str(result.search_docs.get("answer", "")))
        ):
            sources = result.search_docs.get("sources") if result.search_docs else []
            lines = [
                "---",
                "**Scope assessment (search_docs)**",
                "- **In corpus:** no",
                "- **Action:** refused — question outside internal document scope",
            ]
            if sources:
                lines.append(f"- **Retrieved (unrelated):** {', '.join(sources[:4])}")
            if result.grounding_override:
                lines.append("- **Grounding guard:** replaced hallucinated answer with tool refusal")
            return "\n".join(lines)
        return (
            "**Clearance assessment:** lookup_rights was not called — "
            "no confidence score available. Re-run or ask a territory-specific clearance question."
        )

    sources = list(
        dict.fromkeys(
            row.get("source")
            for row in lookup.get("clearance_records", [])
            if row.get("source")
        )
    )
    escalate = lookup.get("needs_escalation") or (
        float(lookup.get("confidence", 0)) < CONFIDENCE_THRESHOLD
    )
    lines = [
        "---",
        "**Clearance assessment (lookup_rights)**",
        f"- **Status:** {lookup.get('status', 'unknown')}",
        f"- **Confidence:** {lookup.get('confidence')}",
        f"- **Territory:** {lookup.get('territory')} → {lookup.get('territory_normalized')}",
        f"- **Escalate to legal:** {'yes' if escalate else 'no'} (threshold {CONFIDENCE_THRESHOLD})",
    ]
    if sources:
        lines.append(f"- **Sources:** {', '.join(sources[:5])}")
    if result.escalation:
        tag = "auto" if result.auto_escalated else "agent"
        lines.append(f"- **Escalation logged ({tag}):** {result.escalation.get('reason', 'n/a')}")
    if result.draft_brief:
        tag = "auto" if result.auto_drafted else "agent"
        lines.append(f"- **Production brief ({tag}):** included in answer above")
    if result.grounding_override:
        lines.append("- **Grounding guard:** answer replaced with search_docs refusal (out of corpus scope)")
    if result.clearance_grounding_override:
        lines.append("- **Clearance grounding guard:** answer aligned with lookup_rights status")
    return "\n".join(lines)


def finalize_agent_result(
    user_message: str,
    result: AgentResult,
    *,
    messages: list | None = None,
) -> AgentResult:
    """Run full post-processing guard pipeline."""
    messages = messages or []
    result = ensure_lookup_rights(user_message, result)
    result = apply_grounding_guard(user_message, result, messages)
    result = apply_clearance_grounding_guard(result)
    result = apply_auto_escalation(result)
    result = apply_auto_draft_brief(user_message, result)
    return result


def human_approval_gate(output: str, auto_approve: bool) -> str | None:
    if auto_approve:
        return output

    print("\n=== PROPOSED OUTPUT (awaiting approval) ===\n")
    print(output)
    print("\n=== END PROPOSED OUTPUT ===")
    decision = input("\nApprove and send? [y/N]: ").strip().lower()
    if decision == "y":
        return output
    print("Output withheld — pending human revision or re-run.")
    return None
