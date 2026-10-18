#!/usr/bin/env python3
"""Add 27 synthetic Harbor Line docs (27 → 54) and merge sharepoint-meta.json."""

from __future__ import annotations

import json
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC = LAB_ROOT / "data" / "synthetic"
META_PATH = SYNTHETIC / "sharepoint-meta.json"

DISCLAIMER = (
    "\n\n---\n*Synthetic document for evaluation and testing only. Not real legal or business records.*\n"
)

NEW_DOCS: list[dict] = [
    {
        "rel": "rights/stock-footage-shutterstock-s01e05.md",
        "meta": {
            "file_name": "stock-footage-shutterstock-s01e05.md",
            "sharepoint_path": "/sites/Legal/Clearances/2026/Stock",
            "department": "legal",
            "confidentiality": "restricted",
            "series": "Harbor Line",
            "doc_type": "rights_clearance",
            "tags": ["stock", "Shutterstock", "S01E05"],
        },
        "body": """# Stock Footage Clearance — Shutterstock Batch HL-STK-905

**Clip package:** Manhattan skyline timelapse (3 clips), license ID SS-4482910.

| Clip | Episode use | Status |
|------|-------------|--------|
| SS-4482910-A | S01E05 cold open | Cleared — web + SVOD |
| SS-4482910-B | S01E05 montage | Cleared — web + SVOD |
| SS-4482910-C | S01E05 reshoot alt | Pending — hold until picture lock |

**Restriction:** No use in trailers without Shutterstock extended promo license (not purchased).

**Contact:** Stock Clearances — legal.stock@harbormedia.example""",
    },
    {
        "rel": "rights/vfx-supervisor-contract-excerpt.md",
        "meta": {
            "file_name": "vfx-supervisor-contract-excerpt.md",
            "sharepoint_path": "/sites/Legal/Contracts/VFX/2026",
            "department": "legal",
            "confidentiality": "highly_restricted",
            "series": "Harbor Line",
            "doc_type": "legal_contract",
            "tags": ["VFX", "contract", "Season2"],
        },
        "body": """# VFX Supervisor Agreement — Excerpt (Harbor Line S02)

**Vendor:** North River VFX Ltd.

- **Scope:** 120 shots across S02E01–E04; delivery ProRes 4444.
- **Credit:** VFX Supervisor credit in main titles sequence 2.
- **Work-for-hire:** All deliverables owned by Harbor Media Group.
- **Insurance:** Vendor maintains $5M E&O through Dec 2027.

**Payment milestone:** 40% on award, 40% on delivery, 20% on broadcast clearance.""",
    },
    {
        "rel": "rights/font-license-graphik-2026.md",
        "meta": {
            "file_name": "font-license-graphik-2026.md",
            "sharepoint_path": "/sites/Legal/Clearances/2026/Fonts",
            "department": "legal",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "rights_clearance",
            "tags": ["fonts", "Graphik", "graphics"],
        },
        "body": """# Font License — Graphik Family (Harbor Line On-Screen Graphics)

**License type:** Broadcast + SVOD — 5 seats, unlimited impressions through series term.

**Allowed:** Lower thirds, episode titles, in-show chyrons.

**Not allowed:** Consumer merchandise, third-party co-brand decks without addendum HL-FNT-12.""",
    },
    {
        "rel": "rights/stunt-insurance-rider-s01e06.md",
        "meta": {
            "file_name": "stunt-insurance-rider-s01e06.md",
            "sharepoint_path": "/sites/Legal/Insurance/2026/Stunts",
            "department": "legal",
            "confidentiality": "restricted",
            "series": "Harbor Line",
            "doc_type": "insurance",
            "tags": ["stunts", "S01E06", "rider"],
        },
        "body": """# Stunt Insurance Rider — Chain Breakaway (S01E06)

**Scene:** HL-106 dock sequence — practical chain breakaway.

**Coverage:** Additional $2M medical aggregate; certified stunt coordinator on set.

**Exclusion:** Night shoot above 20kt wind without re-approval from risk management.""",
    },
    {
        "rel": "legal/privacy-impact-assessment-viewer-data.md",
        "meta": {
            "file_name": "privacy-impact-assessment-viewer-data.md",
            "sharepoint_path": "/sites/Legal/Privacy/2026",
            "department": "legal",
            "confidentiality": "highly_restricted",
            "series": "Harbor Line",
            "doc_type": "privacy_assessment",
            "tags": ["GDPR", "viewer", "analytics"],
        },
        "body": """# DPIA — Harbor Line SVOD Viewer Analytics (EU)

**Processing:** Pseudonymous viewing events for recommendation tuning.

**Lawful basis:** Legitimate interest with opt-out in account settings.

**Retention:** 13 months rolling; no sale to third-party ad networks.""",
    },
    {
        "rel": "legal/sag-aftra-rider-background-s01e08.md",
        "meta": {
            "file_name": "sag-aftra-rider-background-s01e08.md",
            "sharepoint_path": "/sites/Legal/Guild/SAG-AFTRA/2026",
            "department": "legal",
            "confidentiality": "restricted",
            "series": "Harbor Line",
            "doc_type": "guild_rider",
            "tags": ["SAG-AFTRA", "background", "S01E08"],
        },
        "body": """# SAG-AFTRA Background Rider — S01E08 Crowd Scene

**Date:** 2026-11-02 — Ferry terminal exterior.

**Count:** 75 background; 12 minors (permits on file HM-TAL-881).

**Meal penalties:** Avoid — schedule lists 6-hour turnarounds.""",
    },
    {
        "rel": "production/production-brief-s01e08.md",
        "meta": {
            "file_name": "production-brief-s01e08.md",
            "sharepoint_path": "/sites/HarborLine/Production/Briefs/S01E08",
            "department": "production",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "production_brief",
            "tags": ["S01E08", "ferry", "locations"],
        },
        "body": """# Production Brief — Harbor Line S01E08 "Crosscurrent"

**Production code:** HL-108

## Locked locations

1. **Staten Island Ferry — passenger deck** (Hold Nov 18–19, 2026)
2. **Financial District alley — night exteriors** (Hold Nov 21, 2026)

## Open items

- Ferry authority liaison confirming 0600 call time.
- Rain cover: Stage 3 green screen ferry interior.""",
    },
    {
        "rel": "production/production-brief-s01e09.md",
        "meta": {
            "file_name": "production-brief-s01e09.md",
            "sharepoint_path": "/sites/HarborLine/Production/Briefs/S01E09",
            "department": "production",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "production_brief",
            "tags": ["S01E09", "warehouse", "stunts"],
        },
        "body": """# Production Brief — Harbor Line S01E09 "Dead Reckoning"

**Production code:** HL-109

## Notes

- Reuse Brooklyn Navy Yard Building 92 set dressing from S01E04 with redress.
- Stunt review mandatory for forklift practical — see stunt-safety-memo-s01e06 cross-ref.""",
    },
    {
        "rel": "production/transportation-plan-s02.md",
        "meta": {
            "file_name": "transportation-plan-s02.md",
            "sharepoint_path": "/sites/HarborLine/Production/Logistics/2026",
            "department": "production",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "logistics_plan",
            "tags": ["transport", "Season2", "fleet"],
        },
        "body": """# Transportation Plan — Harbor Line Season 2

**Base camp:** Queens soundstage Q7.

**Fleet:** 14 picture vehicles, 6 honey wagons, 2 unit buses.

**Parking holds:** Red Hook lot HM-PARK-22 through Apr 2027.""",
    },
    {
        "rel": "production/catering-allergy-matrix-s01.md",
        "meta": {
            "file_name": "catering-allergy-matrix-s01.md",
            "sharepoint_path": "/sites/HarborLine/Production/Catering/2026",
            "department": "production",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "safety_memo",
            "tags": ["catering", "allergies", "S01"],
        },
        "body": """# Catering Allergy Matrix — Harbor Line S01

**Lead cast:** 2 nut-free zones on set; separate prep table for cast trailer 1.

**Crew notification:** Daily call sheet appendix lists flagged allergens.""",
    },
    {
        "rel": "production/weather-contingency-nov-2026.md",
        "meta": {
            "file_name": "weather-contingency-nov-2026.md",
            "sharepoint_path": "/sites/HarborLine/Production/Schedules/2026",
            "department": "production",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "schedule_memo",
            "tags": ["weather", "November", "exteriors"],
        },
        "body": """# Weather Contingency — November 2026 Exterior Block

**Threshold:** Sustained rain >0.1in/hr cancels East River exteriors.

**Cover sets:** Stage 3 ferry interior, Navy Yard warehouse.""",
    },
    {
        "rel": "post/sound-notes-s01e05.txt",
        "meta": {
            "file_name": "sound-notes-s01e05.txt",
            "sharepoint_path": "/sites/Post/EmailArchive/2026",
            "department": "post_production",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "email",
            "tags": ["sound", "S01E05", "ADR"],
        },
        "body": """From: Supervising Sound Editor
To: Post Producer — Harbor Line

S01E05 — ADR scheduled Dec 2 for lead (dock dialogue). Foley pass approved except crate drop @ 00:41:12 (needs heavier tail).

Music edit: duck theme -3dB under exposition scene 3.""",
    },
    {
        "rel": "post/localization-subtitle-guidelines.md",
        "meta": {
            "file_name": "localization-subtitle-guidelines.md",
            "sharepoint_path": "/sites/Post/Localization/2026",
            "department": "post_production",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "policy",
            "tags": ["subtitles", "EMEA", "LATAM"],
        },
        "body": """# Subtitle Guidelines — Harbor Line International

**Max CPS:** 17 for Latin languages; 15 for CJK where applicable.

**Harbor-specific:** Do not translate "Harbor Brew Co." brand — keep English on-screen.""",
    },
    {
        "rel": "post/qc-report-s01e04-2026.md",
        "meta": {
            "file_name": "qc-report-s01e04-2026.md",
            "sharepoint_path": "/sites/Post/QC/2026",
            "department": "post_production",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "qc_report",
            "tags": ["QC", "S01E04", "SVOD"],
        },
        "body": """# QC Report — S01E04 ProRes Master

**Result:** Conditional pass

**Issues:** 2 frames flash at 00:18:04 (VFX roto); 1 audio pop @ 00:52:11.

**SVOD packaging:** Blocked until VFX pickup delivered Nov 28.""",
    },
    {
        "rel": "distribution/avod-windowing-memo-2026.md",
        "meta": {
            "file_name": "avod-windowing-memo-2026.md",
            "sharepoint_path": "/sites/Distribution/Windowing/2026",
            "department": "distribution",
            "confidentiality": "restricted",
            "series": "Harbor Line",
            "doc_type": "windowing_memo",
            "tags": ["AVOD", "SVOD", "windowing"],
        },
        "body": """# AVOD / SVOD Windowing — Harbor Line S01 (US)

**SVOD exclusive:** 90 days from season drop on Harbor+.

**AVOD:** Episodes 1–3 free with ads after day 91; episodes 4–10 SVOD-only year 1.""",
    },
    {
        "rel": "distribution/brazil-tax-incentive-brief.md",
        "meta": {
            "file_name": "brazil-tax-incentive-brief.md",
            "sharepoint_path": "/sites/Distribution/Sales/LATAM/2026",
            "department": "distribution",
            "confidentiality": "restricted",
            "series": "Harbor Line",
            "doc_type": "sales_brief",
            "tags": ["Brazil", "tax", "copro"],
        },
        "body": """# Brazil Copro Incentive — Harbor Line S02 Scouting

**Status:** Exploratory — no signed copro as of Oct 2026.

**Local partner meetings:** São Paulo Oct 22–24 (distribution only, not financing commit).""",
    },
    {
        "rel": "finance/weekly-cost-report-wk42-2026.md",
        "meta": {
            "file_name": "weekly-cost-report-wk42-2026.md",
            "sharepoint_path": "/sites/Finance/Production/Weekly/2026",
            "department": "finance",
            "confidentiality": "highly_restricted",
            "series": "Harbor Line",
            "doc_type": "cost_report",
            "tags": ["weekly", "EIS", "variance"],
        },
        "body": """# Weekly Cost Report — Week 42 (Harbor Line S01)

**EIS to date:** $41.2M vs approved $42.8M ( favorable 3.7% ).

**Variance drivers:** Location savings East River; overrun in S01E06 stunt rigging.""",
    },
    {
        "rel": "finance/payroll-hold-instructions-s01e07.md",
        "meta": {
            "file_name": "payroll-hold-instructions-s01e07.md",
            "sharepoint_path": "/sites/Finance/Payroll/2026",
            "department": "finance",
            "confidentiality": "highly_restricted",
            "series": "Harbor Line",
            "doc_type": "payroll_memo",
            "tags": ["payroll", "S01E07", "hold"],
        },
        "body": """# Payroll Hold — S01E07 Background Batch B

**Reason:** Missing I-9 copies for 3 performers.

**Release:** Upon HR scan upload to SharePoint /Finance/Payroll/2026/I9.""",
    },
    {
        "rel": "press/embargo-calendar-season2-2026.md",
        "meta": {
            "file_name": "embargo-calendar-season2-2026.md",
            "sharepoint_path": "/sites/Corporate/Comms/Embargoes/2026",
            "department": "communications",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "embargo_calendar",
            "tags": ["embargo", "Season2", "press"],
        },
        "body": """# Embargo Calendar — Harbor Line Season 2

**Cast reveal:** Aug 12, 2026 06:00 ET — lift for trade only until Aug 13 06:00 ET general.

**Trailer:** Sep 30, 2026 coordinated with Harbor+ front page.""",
    },
    {
        "rel": "ad-sales/podcast-integration-rates-2026.md",
        "meta": {
            "file_name": "podcast-integration-rates-2026.md",
            "sharepoint_path": "/sites/AdSales/Rates/2026",
            "department": "ad_sales",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "rate_card",
            "tags": ["podcast", "integration", "CPM"],
        },
        "body": """# Podcast Integration Rate Card — Harbor Line Q4 2026

**Host read (60s):** $85K floor — includes approval of script by legal.

**Donut ad (30s):** $42K — no competitor beverage brands.""",
    },
    {
        "rel": "ad-sales/email-spring-2027-upfront.txt",
        "meta": {
            "file_name": "email-spring-2027-upfront.txt",
            "sharepoint_path": "/sites/AdSales/EmailArchive/2027/Q1",
            "department": "ad_sales",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "email",
            "tags": ["upfront", "2027", "inventory"],
        },
        "body": """Subject: Harbor Line — Spring 2027 upfront hold request

Team,

Please hold 2 premiere adjacency pods for Harbor Line S02 launch week (target Mar 2027). Harbor Brew renewal in negotiation — do not promise category exclusivity until legal signs HL-PP-2027-04.

— Ad Sales Strategy""",
    },
    {
        "rel": "compliance/training-harassment-refresher-2026.md",
        "meta": {
            "file_name": "training-harassment-refresher-2026.md",
            "sharepoint_path": "/sites/Compliance/Training/2026",
            "department": "compliance",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "training",
            "tags": ["HR", "training", "mandatory"],
        },
        "body": """# Mandatory Training — Anti-Harassment Refresher (Cast & Crew)

**Deadline:** Complete by first day on set for S01 block.

**Completion tracking:** Workday course HL-COMP-901.""",
    },
    {
        "rel": "compliance/data-retention-post-production.md",
        "meta": {
            "file_name": "data-retention-post-production.md",
            "sharepoint_path": "/sites/Compliance/Records/2026",
            "department": "compliance",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "records_policy",
            "tags": ["retention", "post", "LTO"],
        },
        "body": """# Records Retention — Post-Production Assets (Harbor Line)

**Camera originals:** 7 years LTO vault.

**Proxies & dailies:** 24 months active NAS, then archive tier.""",
    },
    {
        "rel": "music/spotting-notes-s01e04.md",
        "meta": {
            "file_name": "spotting-notes-s01e04.md",
            "sharepoint_path": "/sites/Music/Spotting/2026",
            "department": "music",
            "confidentiality": "internal",
            "series": "Harbor Line",
            "doc_type": "spotting_notes",
            "tags": ["score", "S01E04", "temp"],
        },
        "body": """# Spotting Notes — S01E04

**Temp track:** Replace licensed indie cue @ warehouse reveal with original score (composer Vasquez).

**Theme:** Full main title not licensed for behind-the-scenes reels — use sting only.""",
    },
    {
        "rel": "music/sync-quote-indie-track-s01e03.md",
        "meta": {
            "file_name": "sync-quote-indie-track-s01e03.md",
            "sharepoint_path": "/sites/Music/Licensing/2026",
            "department": "music",
            "confidentiality": "restricted",
            "series": "Harbor Line",
            "doc_type": "sync_quote",
            "tags": ["indie", "S01E03", "quote"],
        },
        "body": """# Sync Quote — Indie Track "Night Ferry" (S01E03 bar scene)

**Publisher quote:** $18K all-in US linear + SVOD, 5-year term.

**Status:** Pending legal review — alternative library track available at $4K.""",
    },
    {
        "rel": "rights/aerial-drone-permit-nyc-2026.md",
        "meta": {
            "file_name": "aerial-drone-permit-nyc-2026.md",
            "sharepoint_path": "/sites/Legal/Permits/NYC/2026",
            "department": "legal",
            "confidentiality": "restricted",
            "series": "Harbor Line",
            "doc_type": "permit",
            "tags": ["drone", "NYC", "FAA"],
        },
        "body": """# NYC UAV Permit — East River Aerial Establishing (Harbor Line)

**FAA waiver:** Approved Oct 2026 — daylight only, VLOS.

**NYC DCAS film permit:** HM-FILM-UAV-771 — valid Nov 1–30, 2026.

**Restriction:** No flights over Staten Island Ferry routes during active service hours.""",
    },
    {
        "rel": "rights/clearance-summary-s01e04-emea-launch.md",
        "meta": {
            "file_name": "clearance-summary-s01e04-emea-launch.md",
            "sharepoint_path": "/sites/Legal/Clearances/2026/Episodes",
            "department": "legal",
            "confidentiality": "restricted",
            "series": "Harbor Line",
            "doc_type": "rights_clearance",
            "tags": ["clearance summary", "S01E04", "EMEA"],
            "clearance_territories": [
                {"territory": "emea", "platform": "svod", "status": "cleared", "through_date": "2027-12-31"},
                {"territory": "us", "platform": "svod", "status": "cleared", "through_date": "perpetuity"},
            ],
        },
        "body": """# Clearance Summary — S01E04 EMEA SVOD Launch Package

**Episode:** S01E04 "The Warehouse"

| Asset class | Status | Notes |
|-------------|--------|-------|
| Main title theme | Cleared EMEA SVOD through 2027-12-31 | See music-clearance-emea.md |
| Getty archival (dock B-roll) | Cleared | archival-footage-clearance-getty.md |
| Harbor Brew product placement | Cleared S01 only | product-placement-coffee-brand.md |
| Background talent | 3 releases outstanding | talent-release-background-s01.md |

**Packaging:** Approved for EMEA SVOD day-and-date with US, subject to subtitle QC pass.""",
    },
]


def main() -> None:
    if not META_PATH.exists():
        raise SystemExit(f"Missing {META_PATH}")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    existing_names = {d["file_name"] for d in meta.get("documents", [])}

    created = 0
    skipped = 0
    for spec in NEW_DOCS:
        path = SYNTHETIC / spec["rel"]
        fname = spec["meta"]["file_name"]
        if fname in existing_names:
            skipped += 1
            continue
        if path.exists():
            skipped += 1
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(spec["body"].strip() + DISCLAIMER, encoding="utf-8")
        meta.setdefault("documents", []).append(spec["meta"])
        existing_names.add(fname)
        created += 1

    meta["last_synced"] = "2026-10-18T20:00:00Z"
    meta["library"] = meta.get("library", "") + " (expanded catalog)"
    META_PATH.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    total = sum(
        1
        for p in SYNTHETIC.rglob("*")
        if p.is_file() and p.suffix.lower() in {".md", ".txt"} and p.name != "sharepoint-meta.json"
    )
    print(f"Created {created} new files, skipped {skipped}, corpus total {total} docs")


if __name__ == "__main__":
    main()
