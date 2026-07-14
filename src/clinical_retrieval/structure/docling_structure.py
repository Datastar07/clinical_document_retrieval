from __future__ import annotations

"""Build encounters/sections from Docling artifact JSON when available."""

import json
import logging
import re
from pathlib import Path

from clinical_retrieval.schemas import EncounterMeta, PageContent, SectionSpan
from clinical_retrieval.structure.normalizer import extract_dates

logger = logging.getLogger(__name__)

SECTION_MAP = [
    (re.compile(r"chief\s*complaint|history\s+of\s+present|hpi|subjective", re.I), "subjective", "Subjective"),
    (re.compile(r"objective|physical\s+exam|vitals?", re.I), "objective", "Objective"),
    (re.compile(r"assessment|diagnos", re.I), "assessment", "Assessment"),
    (re.compile(r"\bplan\b|orders?", re.I), "plan", "Plan"),
    (re.compile(r"lab(?:oratory)?|hba1c|results", re.I), "laboratory_results", "Laboratory Results"),
    (re.compile(r"medication|meds|drug", re.I), "medications", "Medications"),
    (re.compile(r"referr", re.I), "referrals", "Referrals"),
    (re.compile(r"imaging|ultrasound|x-?ray|mri|ct", re.I), "imaging_orders", "Imaging"),
    (re.compile(r"follow[- ]?up", re.I), "follow_up", "Follow-Up"),
    (re.compile(r"sign(?:ed|ature)", re.I), "signature", "Signature"),
]


def load_docling_artifacts(processed_dir: str | Path) -> dict | None:
    processed_dir = Path(processed_dir)
    summary_path = processed_dir / "docling_summary.json"
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "ok":
        return None
    headings = []
    tables = []
    hp = processed_dir / "docling_headings.json"
    tp = processed_dir / "docling_tables.json"
    if hp.exists():
        headings = json.loads(hp.read_text(encoding="utf-8"))
    if tp.exists():
        tables = json.loads(tp.read_text(encoding="utf-8"))
    md = ""
    mp = processed_dir / "docling_export.md"
    if mp.exists():
        md = mp.read_text(encoding="utf-8")
    return {"summary": summary, "headings": headings, "tables": tables, "markdown": md}


def artifacts_to_encounters(pages: list[PageContent], artifacts: dict) -> list[EncounterMeta]:
    """Fallback encounter segmentation from Visit Date / Date of Service lines."""
    starts: list[tuple[int, str | None, str | None]] = []
    date_pat = re.compile(
        r"(?:Visit Date|Date of Service|DOS)\s*[:\-]\s*(?P<date>[^\n]{6,40})",
        re.I,
    )
    for page in pages:
        m = date_pat.search(page.clean_text)
        if m:
            dates = extract_dates(m.group("date"))
            iso = dates[0] if dates else None
            starts.append((page.page_number, iso, f"DOC-P{page.page_number}"))
    if not starts:
        if not pages:
            return []
        return [
            EncounterMeta(
                encounter_id="DOC-ALL",
                encounter_number=1,
                start_page=pages[0].page_number,
                end_page=pages[-1].page_number,
            )
        ]
    encounters: list[EncounterMeta] = []
    for i, (page_no, iso, eid) in enumerate(starts):
        end = starts[i + 1][0] - 1 if i + 1 < len(starts) else pages[-1].page_number
        encounters.append(
            EncounterMeta(
                encounter_id=eid,
                encounter_number=i + 1,
                encounter_date=iso,
                start_page=page_no,
                end_page=max(page_no, end),
            )
        )
    return encounters


def artifacts_to_sections(
    pages: list[PageContent],
    encounters: list[EncounterMeta],
    artifacts: dict,
) -> list[SectionSpan]:
    page_map = {p.page_number: p for p in pages}
    sections: list[SectionSpan] = []

    # Prefer Docling heading boundaries when present
    headings = artifacts.get("headings") or []
    if headings:
        for i, h in enumerate(headings):
            title = (h.get("text") or "").strip()
            page = h.get("page") or 1
            next_page = headings[i + 1].get("page") if i + 1 < len(headings) else page
            key, heading = "general", title or "General"
            for pat, k, label in SECTION_MAP:
                if pat.search(title):
                    key, heading = k, label
                    break
            enc = _encounter_for_page(encounters, int(page))
            text = ""
            for pn in range(int(page), int(next_page) + 1):
                if pn in page_map:
                    text += page_map[pn].clean_text + "\n"
            sections.append(
                SectionSpan(
                    section=key,
                    section_heading=heading,
                    encounter_id=enc.encounter_id if enc else None,
                    start_page=int(page),
                    end_page=int(next_page),
                    text=text.strip(),
                    char_start=0,
                    char_end=len(text.strip()),
                )
            )
        if sections:
            return sections

    # Fallback: whole-page sections tagged by keywords
    for page in pages:
        enc = _encounter_for_page(encounters, page.page_number)
        key, heading = "general", "Page Content"
        for pat, k, label in SECTION_MAP:
            if pat.search(page.clean_text[:800]):
                key, heading = k, label
                break
        sections.append(
            SectionSpan(
                section=key,
                section_heading=heading,
                encounter_id=enc.encounter_id if enc else None,
                start_page=page.page_number,
                end_page=page.page_number,
                text=page.clean_text,
                char_start=0,
                char_end=len(page.clean_text),
            )
        )
    return sections


def _encounter_for_page(encounters: list[EncounterMeta], page: int) -> EncounterMeta | None:
    for e in encounters:
        if e.start_page <= page <= e.end_page:
            return e
    return encounters[0] if encounters else None
