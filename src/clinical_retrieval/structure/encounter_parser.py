from __future__ import annotations

import re

from dateutil import parser as date_parser

from clinical_retrieval.schemas import EncounterMeta, PageContent

ENCOUNTER_HEAD = re.compile(
    r"Encounter\s+#(?P<num>\d+)\s*\|\s*(?P<eid>ENC-\d+)\s*\|?",
    re.IGNORECASE,
)

TYPE_DATE_LINE = re.compile(
    r"^(?P<etype>.+?)\s*\|\s*(?P<date>[A-Za-z]+\s+\d{1,2},\s+\d{4})(?:\s+\d{1,2}:\d{2})?\s*$",
    re.I,
)
DATE_ONLY_LINE = re.compile(
    r"^(?P<date>[A-Za-z]+\s+\d{1,2},\s+\d{4})(?:\s+\d{1,2}:\d{2})?\s*$",
    re.I,
)

PROVIDER_PATTERN = re.compile(
    r"Provider:\s*(?P<provider>Dr\.\s*[^—\n]+?)(?:\s*—|\s*NPI:|\s*$)",
    re.IGNORECASE,
)


def _normalize_date(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        dt = date_parser.parse(raw, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except (ValueError, OverflowError, TypeError):
        return None


def parse_encounters(pages: list[PageContent]) -> list[EncounterMeta]:
    """Detect encounter boundaries that may span pages."""
    starts: list[tuple[int, dict]] = []

    for page in pages:
        text = page.clean_text
        for m in ENCOUNTER_HEAD.finditer(text):
            info = {
                "encounter_id": m.group("eid"),
                "encounter_number": int(m.group("num")),
                "encounter_type": None,
                "encounter_date": None,
                "provider": None,
                "facility": None,
            }
            after = text[m.end() :]
            # Next lines usually: "<Type> | <Month DD, YYYY> [HH:MM]"
            # Sometimes type ends with "|" and date is on the following line.
            pending_type = None
            for ln in after.splitlines()[:6]:
                ln = ln.strip()
                if not ln or ln == "00:00":
                    continue
                td = TYPE_DATE_LINE.match(ln)
                if td:
                    info["encounter_type"] = td.group("etype").strip(" |")
                    info["encounter_date"] = _normalize_date(td.group("date"))
                    break
                if ln.endswith("|") and pending_type is None and "Provider" not in ln:
                    pending_type = ln.rstrip("|").strip()
                    continue
                d_only = DATE_ONLY_LINE.match(ln)
                if d_only and pending_type:
                    info["encounter_type"] = pending_type
                    info["encounter_date"] = _normalize_date(d_only.group("date"))
                    break
                if pending_type is None and "|" not in ln and "Provider" not in ln:
                    # e.g. lone type line before date
                    pending_type = ln
                    continue
                break

            window = after[:500]
            prov = PROVIDER_PATTERN.search(window)
            if prov:
                info["provider"] = prov.group("provider").strip()

            for ln in window.splitlines()[:12]:
                s = ln.strip()
                if s.endswith("|") or s.startswith("Encounter") or s.startswith("Provider"):
                    continue
                if any(
                    k in s
                    for k in (
                        "Clinic",
                        "Hospital",
                        "Medical Center",
                        "Health Center",
                        "Care Center",
                        "Health",
                    )
                ) and "NPI" not in s and "Follow-Up" not in s:
                    info["facility"] = s
                    break

            starts.append((page.page_number, info))

    if not starts:
        return []

    deduped: list[tuple[int, dict]] = []
    for page_no, info in starts:
        if deduped and deduped[-1][1]["encounter_id"] == info["encounter_id"]:
            continue
        deduped.append((page_no, info))

    last_page = pages[-1].page_number if pages else 1
    encounters: list[EncounterMeta] = []
    for i, (start_page, info) in enumerate(deduped):
        # Include the page where the next encounter begins: prior note often
        # continues (signature / follow-up) before the next heading.
        end_page = deduped[i + 1][0] if i + 1 < len(deduped) else last_page
        encounters.append(
            EncounterMeta(
                encounter_id=info["encounter_id"],
                encounter_number=info["encounter_number"],
                encounter_type=info["encounter_type"],
                encounter_date=info["encounter_date"],
                provider=info["provider"],
                facility=info["facility"],
                start_page=start_page,
                end_page=max(start_page, end_page),
            )
        )
    return encounters


def encounter_for_page(encounters: list[EncounterMeta], page: int) -> EncounterMeta | None:
    for enc in encounters:
        if enc.start_page <= page <= enc.end_page:
            return enc
    return None
