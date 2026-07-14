from __future__ import annotations

import re

from clinical_retrieval.schemas import EncounterMeta, PageContent, SectionSpan
from clinical_retrieval.structure.encounter_parser import encounter_for_page

SECTION_PATTERNS: list[tuple[str, str, re.Pattern[str]]] = [
    ("demographics", "Patient Demographics", re.compile(r"^Patient Demographics\s*$", re.I | re.M)),
    ("allergies", "Allergies", re.compile(r"^Allergies\s*$", re.I | re.M)),
    ("medications", "Current Medications", re.compile(r"^Current Medications\s*$", re.I | re.M)),
    ("medical_history", "Medical & Surgical History", re.compile(r"^Medical\s*&\s*Surgical History\s*$", re.I | re.M)),
    ("social_family", "Social & Family History", re.compile(r"^Social\s*&\s*Family History\s*$", re.I | re.M)),
    ("family_history", "Family History", re.compile(r"^Family History\s*$", re.I | re.M)),
    ("ros", "Review of Systems (ROS)", re.compile(r"^Review of Systems", re.I | re.M)),
    ("subjective", "S — Subjective", re.compile(r"^S\s*[—\-]\s*Subjective", re.I | re.M)),
    ("objective", "O — Objective", re.compile(r"^O\s*[—\-]\s*Objective", re.I | re.M)),
    ("vitals", "Vitals", re.compile(r"^(?:BP|Vitals)\b", re.I | re.M)),
    ("laboratory_results", "Laboratory Results", re.compile(r"^Laboratory Results", re.I | re.M)),
    ("assessment", "A — Assessment / Diagnoses", re.compile(r"^A\s*[—\-]\s*Assessment", re.I | re.M)),
    ("plan", "P — Plan", re.compile(r"^P\s*[—\-]\s*Plan", re.I | re.M)),
    ("laboratory_orders", "Laboratory Orders", re.compile(r"^Laboratory Orders\s*$", re.I | re.M)),
    ("imaging_orders", "Imaging Orders", re.compile(r"^Imaging Orders\s*$", re.I | re.M)),
    ("referrals", "Referrals", re.compile(r"^Referrals\s*$", re.I | re.M)),
    ("clinical_notes", "Clinical Notes", re.compile(r"^Clinical Notes\s*$", re.I | re.M)),
    ("patient_education", "Patient Education", re.compile(r"^Patient Education", re.I | re.M)),
    ("follow_up", "Follow-Up Instructions", re.compile(r"^Follow-Up Instructions\s*$", re.I | re.M)),
    ("signature", "Electronic Signature", re.compile(r"^Electronically signed by:", re.I | re.M)),
]

# Patterns that start a new section (for splitting)
SPLIT_PATTERN = re.compile(
    r"(?=^(?:"
    r"S\s*[—\-]\s*Subjective|"
    r"O\s*[—\-]\s*Objective|"
    r"A\s*[—\-]\s*Assessment|"
    r"P\s*[—\-]\s*Plan|"
    r"Laboratory Results|"
    r"Laboratory Orders|"
    r"Imaging Orders|"
    r"Referrals|"
    r"Clinical Notes|"
    r"Patient Education|"
    r"Follow-Up Instructions|"
    r"Electronically signed by:|"
    r"Encounter\s+#\d+\s*\||"
    r"Patient Demographics|"
    r"Allergies|"
    r"Current Medications|"
    r"Medical\s*&\s*Surgical History|"
    r"Social\s*&\s*Family History|"
    r"Family History|"
    r"Review of Systems"
    r"))",
    re.I | re.M,
)


def _detect_section_label(text: str) -> tuple[str, str]:
    head = text.lstrip()[:240]
    # Prefer explicit SOAP / clinical headings first
    for key, heading, pat in SECTION_PATTERNS:
        if key == "vitals":
            continue
        if pat.search(head):
            return key, heading
    if re.search(r"^P\s*[—\-]\s*Plan", head, re.I | re.M) or head.startswith("•"):
        return "plan", "P — Plan"
    if re.search(r"\b[A-Z]\d{2}(?:\.\d{1,4})?\s*[—\-]", head):
        return "assessment", "A — Assessment / Diagnoses"
    if re.search(r"^(?:BP|Vitals)\b", head, re.I | re.M):
        return "vitals", "Vitals"
    return "general", "General"


def parse_sections(
    pages: list[PageContent],
    encounters: list[EncounterMeta],
) -> list[SectionSpan]:
    """Build section spans by walking cleaned page text with encounter context."""
    # Build per-page encounter map and concatenate with markers
    sections: list[SectionSpan] = []

    # Prefatory pages (before first encounter) as document summary sections
    first_enc_page = encounters[0].start_page if encounters else pages[-1].page_number + 1

    preface_parts: list[tuple[int, str]] = []
    for page in pages:
        if page.page_number >= first_enc_page:
            break
        if page.clean_text.strip():
            preface_parts.append((page.page_number, page.clean_text))

    if preface_parts:
        # Split preface by known headings
        full = "\n\n".join(t for _, t in preface_parts)
        page_for_offset = preface_parts[0][0]
        chunks = [c.strip() for c in SPLIT_PATTERN.split(full) if c and c.strip()]
        if len(chunks) <= 1:
            sections.append(
                SectionSpan(
                    section="patient_summary",
                    section_heading="Patient Summary",
                    encounter_id=None,
                    start_page=preface_parts[0][0],
                    end_page=preface_parts[-1][0],
                    text=full,
                    char_start=0,
                    char_end=len(full),
                )
            )
        else:
            for chunk in chunks:
                key, heading = _detect_section_label(chunk)
                # Approximate page by searching snippet in pages
                start_p = page_for_offset
                end_p = preface_parts[-1][0]
                for pno, ptxt in preface_parts:
                    if chunk[:60] in ptxt or chunk.splitlines()[0][:40] in ptxt:
                        start_p = pno
                        break
                sections.append(
                    SectionSpan(
                        section=key,
                        section_heading=heading,
                        encounter_id=None,
                        start_page=start_p,
                        end_page=end_p,
                        text=chunk,
                        char_start=0,
                        char_end=len(chunk),
                    )
                )

    # Encounter-scoped sections
    for enc in encounters:
        enc_pages = [p for p in pages if enc.start_page <= p.page_number <= enc.end_page]
        if not enc_pages:
            continue
        # Clip text after next encounter heading on last overlapping page is already
        # handled by end_page; still remove trailing next-encounter text if present.
        parts: list[str] = []
        for p in enc_pages:
            text = p.clean_text
            if p.page_number == enc.start_page:
                m = re.search(
                    rf"Encounter\s+#\d+\s*\|\s*{re.escape(enc.encounter_id)}",
                    text,
                    re.I,
                )
                if m:
                    text = text[m.start() :]
            # Trim next encounter if it starts on this page
            nxt = re.search(r"Encounter\s+#\d+\s*\|\s*ENC-\d+", text)
            if nxt and enc.encounter_id not in nxt.group(0):
                # only trim if this match is a different encounter
                m2 = re.search(r"Encounter\s+#\d+\s*\|\s*(ENC-\d+)", text[nxt.start() :])
                if m2 and m2.group(1) != enc.encounter_id:
                    text = text[: nxt.start()]
            parts.append(text)

        full = "\n\n".join(parts).strip()
        pieces = [c.strip() for c in SPLIT_PATTERN.split(full) if c and c.strip()]
        if not pieces:
            pieces = [full]

        for piece in pieces:
            key, heading = _detect_section_label(piece)
            # Estimate page range
            start_p, end_p = enc.start_page, enc.end_page
            snippet = piece.splitlines()[0][:50] if piece else ""
            for p in enc_pages:
                if snippet and snippet in p.clean_text:
                    start_p = p.page_number
                    break
            for p in reversed(enc_pages):
                tail = piece[-80:] if len(piece) > 80 else piece
                if tail and tail in p.clean_text:
                    end_p = p.page_number
                    break

            sections.append(
                SectionSpan(
                    section=key,
                    section_heading=heading,
                    encounter_id=enc.encounter_id,
                    start_page=start_p,
                    end_page=max(start_p, end_p),
                    text=piece,
                    char_start=0,
                    char_end=len(piece),
                )
            )

    return sections
