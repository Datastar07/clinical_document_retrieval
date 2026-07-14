"""Built-in structure parsers for different clinical document sources."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from clinical_retrieval.schemas import EncounterMeta, PageContent, SectionSpan
from clinical_retrieval.structure.base import register_parser
from clinical_retrieval.structure.docling_structure import (
    artifacts_to_encounters,
    artifacts_to_sections,
    load_docling_artifacts,
)
from clinical_retrieval.structure.encounter_parser import parse_encounters as _parse_encounters_soap
from clinical_retrieval.structure.normalizer import extract_dates
from clinical_retrieval.structure.section_parser import parse_sections as _parse_sections_soap

logger = logging.getLogger(__name__)

SOAP_MARKERS = re.compile(
    r"(?:Encounter\s*#\s*\d+|ENC-\d+|"
    r"S\s*[—\-·]\s*Subjective|O\s*[—\-·]\s*Objective|"
    r"A\s*[—\-·]\s*Assessment|P\s*[—\-·]\s*Plan)",
    re.I,
)

PROVIDER_CREDS = re.compile(
    r"(?:Dr\.\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+,\s*(?:MD|DO|NP|PA|RN|APRN)\b"
)


def _page_structure_text(page: PageContent) -> str:
    """Prefer raw page text for markers (furniture strip can drop ENC headers)."""
    return f"{page.text or ''}\n{page.clean_text or ''}"


def has_soap_markers(pages: list[PageContent], sample_pages: int = 8) -> bool:
    sample = "\n".join(_page_structure_text(p) for p in pages[:sample_pages])
    hits = len(SOAP_MARKERS.findall(sample))
    return hits >= 2


def _with_structure_text(pages: list[PageContent]) -> list[PageContent]:
    """Copy pages with clean_text augmented for encounter/section regex parsers."""
    out: list[PageContent] = []
    for p in pages:
        merged = _page_structure_text(p).strip()
        if merged == (p.clean_text or "").strip():
            out.append(p)
            continue
        out.append(p.model_copy(update={"clean_text": merged}))
    return out


class SyntheticSoapParser:
    """Regex/layout parser tuned to ENC-/SOAP chart layouts."""

    name = "synthetic_soap"

    def parse_encounters(self, pages: list[PageContent]) -> list[EncounterMeta]:
        return _parse_encounters_soap(_with_structure_text(pages))

    def parse_sections(
        self,
        pages: list[PageContent],
        encounters: list[EncounterMeta],
    ) -> list[SectionSpan]:
        return _parse_sections_soap(_with_structure_text(pages), encounters)


class GenericStructureParser:
    """Heuristics for Progress Note / Visit Date / table-heavy / non-SOAP layouts."""

    name = "generic"

    ENCOUNTER_PATTERNS = [
        re.compile(
            r"(?:Visit Date|Date of Service|DOS)\s*[:\-]\s*(?P<date>[^\n]{6,40})",
            re.I,
        ),
        re.compile(
            r"\b(?:Encounter|Office Visit|Progress Note)\b\s*"
            r"(?:#?\d+\s*)?(?:\|\s*)?(?P<eid>ENC-?\d+|[A-Z]{2,}-\d+)?",
            re.I,
        ),
        re.compile(
            r"\b(?P<etype>Annual Physical|Follow-?Up Visit|Office Visit|Telehealth|"
            r"Urgent Care|Progress Note|Ambulatory Visit)\b.{0,60}?"
            r"(?P<date>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}"
            r"|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})",
            re.I,
        ),
    ]

    SECTION_HEADINGS = [
        ("subjective", "Subjective", re.compile(r"^(?:S\s*[—\-]?\s*)?Subjective\b|^HPI\b|^History of Present Illness\b|^Chief Complaint\b", re.I | re.M)),
        ("objective", "Objective", re.compile(r"^(?:O\s*[—\-]?\s*)?Objective\b|^Physical Exam\b|^Vitals?\b", re.I | re.M)),
        ("assessment", "Assessment", re.compile(r"^(?:A\s*[—\-]?\s*)?Assessment\b|^Diagnos(?:is|es)\b", re.I | re.M)),
        ("plan", "Plan", re.compile(r"^(?:P\s*[—\-]?\s*)?Plan\b|^Orders?\b", re.I | re.M)),
        ("medications", "Medications", re.compile(r"^(?:Current\s+|Active\s+)?Medications?\b", re.I | re.M)),
        ("allergies", "Allergies", re.compile(r"^Allergies\b", re.I | re.M)),
        ("laboratory_results", "Laboratory Results", re.compile(r"^Lab(?:oratory)?\s*(?:Results?|Values?)?\b|^Labs\b", re.I | re.M)),
        ("referrals", "Referrals", re.compile(r"^Referrals?\b", re.I | re.M)),
        ("imaging_orders", "Imaging", re.compile(r"^Imaging(?:\s+Orders?)?\b", re.I | re.M)),
        ("follow_up", "Follow-Up", re.compile(r"^Follow[- ]?Up\b", re.I | re.M)),
        ("signature", "Signature", re.compile(r"Electronically signed by:|Signed by:", re.I | re.M)),
    ]

    def parse_encounters(self, pages: list[PageContent]) -> list[EncounterMeta]:
        pages = _with_structure_text(pages)
        # Only prefer SOAP when markers are strong — avoid forcing soap on progress notes
        if has_soap_markers(pages):
            soap = _parse_encounters_soap(pages)
            if soap and any(e.encounter_id and e.encounter_id.startswith("ENC") for e in soap):
                return soap

        starts: list[tuple[int, dict]] = []
        for page in pages:
            text = page.clean_text
            matched = False
            for pat in self.ENCOUNTER_PATTERNS:
                m = pat.search(text)
                if not m:
                    continue
                gd = m.groupdict()
                date_raw = gd.get("date")
                iso = None
                if date_raw:
                    dates = extract_dates(date_raw)
                    iso = dates[0] if dates else None
                eid = gd.get("eid")
                if eid:
                    if eid.startswith("ENC") and not eid.startswith("ENC-"):
                        eid = eid.replace("ENC", "ENC-", 1)
                else:
                    eid = f"GEN-P{page.page_number}"
                    if iso:
                        eid = f"VISIT-{iso}"
                provider = None
                pm = PROVIDER_CREDS.search(text)
                if pm:
                    provider = pm.group(0)
                starts.append(
                    (
                        page.page_number,
                        {
                            "encounter_id": eid,
                            "encounter_type": gd.get("etype") or "Clinical Visit",
                            "encounter_date": iso,
                            "provider": provider,
                            "facility": None,
                        },
                    )
                )
                matched = True
                break
            if not matched and PROVIDER_CREDS.search(text) and re.search(
                r"(?:Progress Note|Office Visit|Clinical Note)", text, re.I
            ):
                starts.append(
                    (
                        page.page_number,
                        {
                            "encounter_id": f"GEN-P{page.page_number}",
                            "encounter_type": "Progress Note",
                            "encounter_date": None,
                            "provider": PROVIDER_CREDS.search(text).group(0),
                            "facility": None,
                        },
                    )
                )

        if not starts:
            if not pages:
                return []
            return [
                EncounterMeta(
                    encounter_id="GEN-DOC",
                    encounter_number=1,
                    encounter_type="Document",
                    encounter_date=None,
                    provider=None,
                    facility=None,
                    start_page=pages[0].page_number,
                    end_page=pages[-1].page_number,
                )
            ]

        deduped: list[tuple[int, dict]] = []
        for page_no, info in starts:
            if deduped and deduped[-1][1]["encounter_id"] == info["encounter_id"]:
                continue
            deduped.append((page_no, info))

        last = pages[-1].page_number
        out: list[EncounterMeta] = []
        for i, (start, info) in enumerate(deduped):
            end = deduped[i + 1][0] - 1 if i + 1 < len(deduped) else last
            out.append(
                EncounterMeta(
                    encounter_id=info["encounter_id"],
                    encounter_number=i + 1,
                    encounter_type=info.get("encounter_type"),
                    encounter_date=info.get("encounter_date"),
                    provider=info.get("provider"),
                    facility=info.get("facility"),
                    start_page=start,
                    end_page=max(start, end),
                )
            )
        return out

    def parse_sections(
        self,
        pages: list[PageContent],
        encounters: list[EncounterMeta],
    ) -> list[SectionSpan]:
        pages = _with_structure_text(pages)
        if has_soap_markers(pages):
            soap = _parse_sections_soap(pages, encounters)
            if soap:
                return soap

        sections: list[SectionSpan] = []
        for enc in encounters:
            enc_pages = [p for p in pages if enc.start_page <= p.page_number <= enc.end_page]
            full = "\n\n".join(p.clean_text for p in enc_pages)
            found = False
            for key, heading, pat in self.SECTION_HEADINGS:
                matches = list(pat.finditer(full))
                if not matches:
                    continue
                found = True
                for idx, m in enumerate(matches):
                    start = m.start()
                    end = matches[idx + 1].start() if idx + 1 < len(matches) else len(full)
                    for _, _, other in self.SECTION_HEADINGS:
                        if other is pat:
                            continue
                        nxt = other.search(full, start + 1)
                        if nxt and nxt.start() < end:
                            end = nxt.start()
                    text = full[start:end].strip()
                    if text:
                        sections.append(
                            SectionSpan(
                                section=key,
                                section_heading=heading,
                                encounter_id=enc.encounter_id,
                                start_page=enc.start_page,
                                end_page=enc.end_page,
                                text=text,
                                char_start=0,
                                char_end=len(text),
                            )
                        )
            if not found and full.strip():
                sections.append(
                    SectionSpan(
                        section="general",
                        section_heading="General",
                        encounter_id=enc.encounter_id,
                        start_page=enc.start_page,
                        end_page=enc.end_page,
                        text=full.strip(),
                        char_start=0,
                        char_end=len(full.strip()),
                    )
                )
        return sections


class DoclingStructureParser:
    """Consume Docling headings/tables when artifacts are present."""

    name = "docling"

    def __init__(self, processed_dir: str | Path | None = None):
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self._artifacts: dict | None = None

    def set_processed_dir(self, path: str | Path) -> None:
        self.processed_dir = Path(path)
        self._artifacts = None

    def _load(self) -> dict | None:
        if self._artifacts is not None:
            return self._artifacts
        if not self.processed_dir:
            return None
        self._artifacts = load_docling_artifacts(self.processed_dir)
        return self._artifacts

    def parse_encounters(self, pages: list[PageContent]) -> list[EncounterMeta]:
        arts = self._load()
        if arts and (arts.get("headings") or arts.get("tables") or arts.get("markdown")):
            enc = artifacts_to_encounters(pages, arts)
            if enc:
                logger.info("DoclingStructureParser: %d encounters from artifacts", len(enc))
                return enc
        if has_soap_markers(pages):
            return SyntheticSoapParser().parse_encounters(pages)
        return GenericStructureParser().parse_encounters(pages)

    def parse_sections(
        self,
        pages: list[PageContent],
        encounters: list[EncounterMeta],
    ) -> list[SectionSpan]:
        arts = self._load()
        if arts:
            secs = artifacts_to_sections(pages, encounters, arts)
            if secs:
                return secs
        if has_soap_markers(pages):
            return SyntheticSoapParser().parse_sections(pages, encounters)
        return GenericStructureParser().parse_sections(pages, encounters)


class AutoStructureParser:
    """SOAP markers → soap; else Docling ok → docling; else generic."""

    name = "auto"

    def __init__(self, processed_dir: str | Path | None = None):
        self.processed_dir = Path(processed_dir) if processed_dir else None
        self.selected: str | None = None

    def set_processed_dir(self, path: str | Path) -> None:
        self.processed_dir = Path(path)

    def _choose(self, pages: list[PageContent]) -> str:
        if has_soap_markers(pages):
            return "synthetic_soap"
        arts = load_docling_artifacts(self.processed_dir) if self.processed_dir else None
        if arts and arts.get("summary", {}).get("status") == "ok":
            if arts.get("headings") or arts.get("tables") or arts.get("markdown"):
                return "docling"
        return "generic"

    def parse_encounters(self, pages: list[PageContent]) -> list[EncounterMeta]:
        choice = self._choose(pages)
        self.selected = choice
        logger.info("AutoStructureParser selected: %s", choice)
        if choice == "docling":
            p = DoclingStructureParser(self.processed_dir)
            return p.parse_encounters(pages)
        return get_parser_by_name(choice).parse_encounters(pages)

    def parse_sections(
        self,
        pages: list[PageContent],
        encounters: list[EncounterMeta],
    ) -> list[SectionSpan]:
        choice = self.selected or self._choose(pages)
        self.selected = choice
        if choice == "docling":
            p = DoclingStructureParser(self.processed_dir)
            return p.parse_sections(pages, encounters)
        return get_parser_by_name(choice).parse_sections(pages, encounters)


def get_parser_by_name(name: str):
    from clinical_retrieval.structure.base import get_parser

    return get_parser(name)


# Register instances (not classes)
register_parser(SyntheticSoapParser())
register_parser(GenericStructureParser())
register_parser(DoclingStructureParser())
register_parser(AutoStructureParser())
