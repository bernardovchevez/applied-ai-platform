"""Build structured clearance territory rows for SQL warehouse (Option C)."""

from __future__ import annotations

import re

# Map markdown / memo status phrases → canonical clearance status
STATUS_NORMALIZE = {
    "cleared": "cleared",
    "pending": "pending",
    "not cleared": "pending",
    "excluded": "excluded",
    "unknown": "unknown",
}

TAG_TERRITORY_DEFAULTS: dict[str, tuple[str, str]] = {
    "emea": ("emea", "cleared"),
    "latam": ("latam", "cleared"),
    "apac": ("apac", "pending"),
}


def normalize_row_status(raw: str) -> str:
    lowered = raw.lower().strip().strip("*")
    if lowered in STATUS_NORMALIZE:
        return STATUS_NORMALIZE[lowered]
    if "excluded" in lowered:
        return "excluded"
    if "not cleared" in lowered or "pending" in lowered:
        return "pending"
    if "cleared" in lowered:
        return "cleared"
    return "unknown"


def rows_from_document_meta(meta: dict) -> list[dict]:
    """Expand SharePoint warehouse metadata into clearance_territories rows."""
    if meta.get("doc_type") != "rights_clearance":
        return []

    series = meta.get("series") or "Harbor Line"
    source = meta.get("source") or meta.get("file_name") or ""
    sharepoint_path = meta.get("sharepoint_path")
    rows: list[dict] = []
    seen: set[str] = set()

    def add_row(
        territory: str,
        status: str,
        *,
        platform: str | None = None,
        through_date: str | None = None,
    ) -> None:
        terr = territory.lower()
        if terr in seen:
            return
        seen.add(terr)
        rows.append(
            {
                "series": series,
                "territory": terr,
                "platform": platform,
                "status": normalize_row_status(status),
                "through_date": through_date,
                "source": source,
                "sharepoint_path": sharepoint_path,
            }
        )

    for item in meta.get("clearance_territories") or []:
        if not isinstance(item, dict):
            continue
        territory = item.get("territory")
        status = item.get("status")
        if territory and status:
            add_row(
                str(territory),
                str(status),
                platform=item.get("platform"),
                through_date=item.get("through_date"),
            )

    territory_status = meta.get("territory_status") or {}
    if isinstance(territory_status, dict):
        for territory, status in territory_status.items():
            add_row(str(territory), str(status))

    tags = [str(t).lower() for t in (meta.get("tags") or [])]
    for tag, (territory, status) in TAG_TERRITORY_DEFAULTS.items():
        if tag in tags and territory not in seen:
            add_row(territory, status, platform="svod")

    return rows


def parse_clearance_table(text: str, *, series: str, source: str, sharepoint_path: str | None) -> list[dict]:
    """Parse markdown clearance tables when warehouse JSON is absent."""
    rows: list[dict] = []
    in_table = False
    for line in text.splitlines():
        if re.match(r"^\|\s*Territory", line, re.I):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.strip().startswith("|"):
            break
        if re.match(r"^\|\s*-+", line):
            continue
        cells = [c.strip().strip("*") for c in line.split("|")[1:-1]]
        if len(cells) < 3:
            continue
        territory_raw, platform, status_raw = cells[0], cells[1], cells[2]
        through_date = cells[3] if len(cells) > 3 else None
        territory = _territory_from_cell(territory_raw)
        if not territory:
            continue
        rows.append(
            {
                "series": series,
                "territory": territory,
                "platform": platform.lower() if platform else None,
                "status": normalize_row_status(status_raw),
                "through_date": through_date if through_date and through_date != "—" else None,
                "source": source,
                "sharepoint_path": sharepoint_path,
            }
        )
    return rows


def _territory_from_cell(raw: str) -> str | None:
    text = raw.lower()
    mapping = (
        ("emea", "emea"),
        ("latin america", "latam"),
        ("apac", "apac"),
        ("china", "china"),
        ("japan", "japan"),
        ("south korea", "korea"),
        ("korea", "korea"),
        ("australia", "apac"),
        ("southeast asia", "apac"),
        ("united states", "us"),
    )
    for needle, code in mapping:
        if needle in text:
            return code
    return None


def build_clearance_rows(documents: list[dict]) -> list[dict]:
    """Collect all territory rows from loaded corpus documents."""
    rows: list[dict] = []
    for doc in documents:
        meta = doc["metadata"]
        meta_rows = rows_from_document_meta(meta)
        if meta_rows:
            rows.extend(meta_rows)
            continue
        if meta.get("doc_type") == "rights_clearance":
            parsed = parse_clearance_table(
                doc["text"],
                series=meta.get("series") or "Harbor Line",
                source=meta.get("source") or "",
                sharepoint_path=meta.get("sharepoint_path"),
            )
            rows.extend(parsed)
    return rows
