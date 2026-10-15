"""ReAct sub-agents for LangGraph workflow nodes, with deterministic fallbacks."""

from __future__ import annotations

from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

from tools import (
    CONFIDENCE_THRESHOLD,
    draft_brief,
    escalate_legal,
    lookup_rights,
    search_docs,
)
from workflow import (
    extract_tool_summaries,
    get_llm,
    infer_series_from_message,
    infer_territory_from_message,
    is_clearance_question,
)

RESEARCHER_PROMPT = (
    "You are the Researcher agent for a media operations team. "
    "Call search_docs to answer the user question from internal documents. "
    "Do not answer from general knowledge. Cite source filenames."
)

COMPLIANCE_PROMPT = (
    "You are the Compliance agent for a media operations team. "
    "For territory clearance questions, call lookup_rights first. "
    "If clearance is pending, unknown, excluded, or confidence is below 0.7, call escalate_legal. "
    "Your final message must match tool JSON results."
)

PRODUCER_PROMPT = (
    "You are the Producer agent for a media operations team. "
    "Call draft_brief with the user topic and gathered context from the message. "
    "Include clearance and research context in the context argument."
)


def _invoke_react(tools: list, prompt: str, user_message: str, *, max_turns: int = 6) -> list:
    agent = create_react_agent(get_llm(), tools, prompt=prompt)
    result = agent.invoke(
        {"messages": [HumanMessage(content=user_message)]},
        config={"recursion_limit": max_turns},
    )
    return result.get("messages", [])


def run_researcher(user_message: str, *, use_react: bool = True) -> dict:
    """Researcher node: ReAct over search_docs with deterministic fallback."""
    from tools import search_docs_impl_for_guard, search_web_stub_impl

    if use_react:
        _invoke_react([search_docs], RESEARCHER_PROMPT, user_message, max_turns=5)

    # Authoritative corpus search; ReAct may paraphrase or skip the tool on small models.
    search = search_docs_impl_for_guard(user_message)
    return {
        "search_docs": search,
        "web_search": search_web_stub_impl(user_message),
        "researcher_mode": "react" if use_react else "deterministic",
    }


def run_compliance(
    user_message: str,
    *,
    search_docs_result: dict | None = None,
    use_react: bool = True,
) -> dict:
    """Compliance node: ReAct over lookup_rights + escalate_legal with fallbacks."""
    from tools import (
        check_ip_flags_impl,
        compliance_llm_review_impl,
        escalate_legal_impl_for_guard,
        lookup_rights_impl_for_guard,
        should_scan_ip_flags,
    )

    series = infer_series_from_message(user_message)
    lookup = None
    escalation = None

    if use_react and is_clearance_question(user_message):
        context = user_message
        if search_docs_result:
            context += f"\n\nPrior research:\n{search_docs_result.get('answer', '')[:800]}"
        messages = _invoke_react(
            [lookup_rights, escalate_legal],
            COMPLIANCE_PROMPT,
            context,
            max_turns=6,
        )
        _, _, escalation = extract_tool_summaries(messages)

    # Authoritative lookup from inferred territory; ReAct may pass wrong args on small models.
    if is_clearance_question(user_message):
        territory = infer_territory_from_message(user_message)
        if territory:
            lookup = lookup_rights_impl_for_guard(series, territory)

    updates: dict = {"lookup_rights": lookup, "escalation": escalation}

    if should_scan_ip_flags(user_message):
        updates["ip_flags"] = check_ip_flags_impl(series, user_message)

    notes = compliance_llm_review_impl(
        user_message,
        lookup=lookup,
        search=search_docs_result,
        ip_flags=updates.get("ip_flags"),
    )
    if notes:
        updates["compliance_notes"] = notes

    if lookup and (lookup.get("needs_escalation") or float(lookup.get("confidence", 1)) < CONFIDENCE_THRESHOLD):
        if not escalation:
            reason = (
                f"{lookup.get('series', 'series')} clearance in "
                f"{lookup.get('territory_normalized') or lookup.get('territory')} is "
                f"{lookup.get('status')} (confidence {lookup.get('confidence')}), "
                f"below threshold {CONFIDENCE_THRESHOLD}"
            )
            updates["escalation"] = escalate_legal_impl_for_guard(
                reason, float(lookup.get("confidence", 0))
            )
            updates["auto_escalated"] = True

    updates["compliance_mode"] = "react" if use_react and lookup else "deterministic"
    return updates


def run_producer(user_message: str, context: str, *, use_react: bool = True) -> dict:
    """Producer node: ReAct over draft_brief with deterministic output."""
    from tools import draft_brief_impl_for_guard

    if use_react:
        prompt = f"{user_message}\n\nContext for draft_brief:\n{context}"
        _invoke_react([draft_brief], PRODUCER_PROMPT, prompt, max_turns=5)

    brief = draft_brief_impl_for_guard(user_message, context)
    return {
        "draft_brief": brief,
        "auto_drafted": True,
        "producer_mode": "react" if use_react else "deterministic",
    }


def run_critic(
    user_message: str,
    draft: str,
    *,
    lookup: dict | None = None,
    search: dict | None = None,
    critic_loops: int = 0,
    max_revision_loops: int = 1,
    use_llm: bool = True,
) -> dict:
    """Critic/reviewer: deterministic QA + optional LLM review on draft vs tool results."""
    from tools import critic_llm_review_impl, parse_critic_llm_concerns
    from workflow import (
        answer_contradicts_lookup,
        claims_cleared,
        claims_pending,
        is_refusal,
        is_summary_flow,
        search_indicates_out_of_scope,
    )

    issues: list[str] = []
    text = (draft or "").strip()

    if not text:
        issues.append("No draft or answer to review")

    if lookup and text and answer_contradicts_lookup(text, lookup):
        status = lookup.get("status", "unknown")
        issues.append(f"Draft contradicts lookup_rights status ({status!r})")

    if search and (
        is_refusal(str(search.get("answer", "")))
        or search_indicates_out_of_scope(str(search.get("answer", "")))
    ) and text and not is_refusal(text):
        issues.append("Draft should reflect corpus refusal from search_docs")

    if lookup and text:
        status = lookup.get("status")
        if status in ("pending", "excluded", "unknown") and claims_cleared(text):
            issues.append(f"Draft claims cleared but lookup status is {status!r}")
        if status == "cleared" and claims_pending(text) and not claims_cleared(text):
            issues.append("Draft claims pending but lookup status is cleared")

    if is_summary_flow(user_message) and text:
        if len(text) < 120:
            issues.append("Production brief is too short for a summary request")
        known_sources = [
            row.get("source")
            for row in (lookup or {}).get("clearance_records", [])
            if row.get("source")
        ]
        known_sources.extend(search.get("sources") or [] if search else [])
        unique_sources = list(dict.fromkeys(known_sources))
        if unique_sources and not any(src.lower() in text.lower() for src in unique_sources):
            issues.append("Brief missing source filenames from lookup/search")

    revision_issues = {
        "Production brief is too short for a summary request",
        "Brief missing source filenames from lookup/search",
    }
    needs_revision = (
        is_summary_flow(user_message)
        and critic_loops < max_revision_loops
        and any(issue in revision_issues for issue in issues)
    )
    approved = not issues

    llm_notes: str | None = None
    llm_concerns: list[str] = []
    if use_llm:
        llm_notes = critic_llm_review_impl(
            user_message,
            text,
            lookup=lookup,
            search=search,
            deterministic_issues=issues,
        )
        llm_concerns = parse_critic_llm_concerns(llm_notes)

    return {
        "critic_review": {
            "approved": approved and not llm_concerns,
            "needs_revision": needs_revision,
            "issues": issues,
            "llm_notes": llm_notes,
            "llm_concerns": llm_concerns,
            "draft_chars": len(text),
        },
        "critic_loops": critic_loops + 1,
        **({"draft_brief": None} if needs_revision else {}),
    }
