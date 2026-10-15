"""Clearance status model shared by lookup_rights and Compliance agent."""

from __future__ import annotations

CONFIDENCE_THRESHOLD = 0.7

CLEARANCE_STATUSES = frozenset({"cleared", "pending", "excluded", "unknown"})

CONFIDENCE_BY_STATUS: dict[str, float] = {
    "cleared": 0.9,
    "pending": 0.35,
    "excluded": 0.9,
    "unknown": 0.5,
}

# Most restrictive status wins when multiple memos match.
STATUS_PRIORITY = ("excluded", "pending", "unknown", "cleared")


def resolve_clearance_status(
    territory_row_statuses: list[str],
    doc_tiers: set[str],
    *,
    has_territory_match: bool,
) -> tuple[str, float]:
    """Derive status + confidence from warehouse row overrides or doc-level tiers."""
    for status in STATUS_PRIORITY:
        if status in territory_row_statuses:
            return status, CONFIDENCE_BY_STATUS[status]

    if not has_territory_match:
        return "unknown", CONFIDENCE_BY_STATUS["unknown"]

    if "excluded" in doc_tiers:
        return "excluded", CONFIDENCE_BY_STATUS["excluded"]
    if "pending" in doc_tiers:
        return "pending", CONFIDENCE_BY_STATUS["pending"]
    if "cleared" in doc_tiers:
        return "cleared", CONFIDENCE_BY_STATUS["cleared"]
    return "unknown", CONFIDENCE_BY_STATUS["unknown"]
