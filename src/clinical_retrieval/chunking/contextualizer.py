from __future__ import annotations

from clinical_retrieval.schemas import BBox, Chunk, ChunkMetadata, EncounterMeta, PageContent, SectionSpan, TextBlock
from clinical_retrieval.structure.entity_extractor import extract_entities
from clinical_retrieval.structure.normalizer import expand_aliases, normalize_text


def format_date_human(iso: str | None) -> str:
    if not iso:
        return "Unknown"
    try:
        y, m, d = iso.split("-")
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        return f"{months[int(m)-1]} {int(d)}, {y}"
    except Exception:
        return iso


def build_contextual_header(
    *,
    patient_name: str,
    encounter: EncounterMeta | None,
    section_heading: str | None,
    page_start: int,
    page_end: int,
) -> str:
    lines = [f"Patient: {patient_name}"]
    if encounter:
        if encounter.encounter_type:
            lines.append(f"Encounter: {encounter.encounter_type}")
        if encounter.encounter_id:
            lines.append(f"Encounter ID: {encounter.encounter_id}")
        if encounter.encounter_date:
            lines.append(
                f"Encounter Date: {format_date_human(encounter.encounter_date)}"
            )
        if encounter.provider:
            lines.append(f"Provider: {encounter.provider}")
        if encounter.facility:
            lines.append(f"Facility: {encounter.facility}")
    if section_heading:
        lines.append(f"Section: {section_heading}")
    if page_start == page_end:
        lines.append(f"Page: {page_start}")
    else:
        lines.append(f"Pages: {page_start}-{page_end}")
    return "\n".join(lines)


def _blocks_for_pages(
    pages_by_num: dict[int, PageContent] | None,
    page_start: int,
    page_end: int,
) -> list[tuple[int, TextBlock]]:
    if not pages_by_num:
        return []
    out: list[tuple[int, TextBlock]] = []
    for p in range(page_start, page_end + 1):
        page = pages_by_num.get(p)
        if not page:
            continue
        for b in page.blocks:
            out.append((p, b))
    return out


def resolve_bounding_boxes(
    raw_text: str,
    *,
    page_start: int,
    page_end: int,
    pages_by_num: dict[int, PageContent] | None = None,
    max_boxes: int = 8,
) -> list[BBox]:
    """Map chunk text onto PyMuPDF block bboxes (immutable grounding)."""
    blocks = _blocks_for_pages(pages_by_num, page_start, page_end)
    if not blocks:
        # Fallback page-level box if we at least know page geometry
        if pages_by_num and page_start in pages_by_num:
            page = pages_by_num[page_start]
            return [
                BBox(
                    page=page_start,
                    x0=0.0,
                    y0=0.0,
                    x1=float(page.width or 0.0),
                    y1=float(page.height or 0.0),
                )
            ]
        return []

    needles = [ln.strip() for ln in raw_text.splitlines() if len(ln.strip()) >= 12][:12]
    if not needles:
        needles = [raw_text.strip()[:80]] if raw_text.strip() else []

    hits: list[BBox] = []
    seen: set[tuple[int, float, float, float, float]] = set()
    for needle in needles:
        nl = needle.lower()
        for page_no, block in blocks:
            bt = (block.text or "").lower()
            if not bt:
                continue
            if nl in bt or bt in nl or (len(nl) > 20 and nl[:40] in bt):
                if len(block.bbox) >= 4:
                    key = (
                        page_no,
                        round(block.bbox[0], 1),
                        round(block.bbox[1], 1),
                        round(block.bbox[2], 1),
                        round(block.bbox[3], 1),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    hits.append(
                        BBox(
                            page=page_no,
                            x0=float(block.bbox[0]),
                            y0=float(block.bbox[1]),
                            x1=float(block.bbox[2]),
                            y1=float(block.bbox[3]),
                        )
                    )
                    if len(hits) >= max_boxes:
                        return hits
    if hits:
        return hits

    # Union of largest blocks on first page of span
    page_blocks = [(p, b) for p, b in blocks if p == page_start]
    page_blocks.sort(key=lambda x: len(x[1].text or ""), reverse=True)
    for page_no, block in page_blocks[:3]:
        if len(block.bbox) >= 4:
            hits.append(
                BBox(
                    page=page_no,
                    x0=float(block.bbox[0]),
                    y0=float(block.bbox[1]),
                    x1=float(block.bbox[2]),
                    y1=float(block.bbox[3]),
                )
            )
    return hits


def contextualize_chunk(
    raw_text: str,
    *,
    patient_name: str,
    patient_id: str,
    document_id: str,
    source_document: str,
    encounter: EncounterMeta | None,
    section: SectionSpan | None,
    page_start: int,
    page_end: int,
    chunk_id: str,
    chunk_type: str,
    table_type: str | None = None,
    char_start: int = 0,
    char_end: int = 0,
    pages_by_num: dict[int, PageContent] | None = None,
    image_path: str | None = None,
) -> Chunk:
    heading = section.section_heading if section else None
    section_key = section.section if section else None
    header = build_contextual_header(
        patient_name=patient_name,
        encounter=encounter,
        section_heading=heading,
        page_start=page_start,
        page_end=page_end,
    )
    retrieval_text = f"{header}\n\n{raw_text}"
    normalized = expand_aliases(raw_text)
    entities = extract_entities(raw_text)
    if encounter and encounter.encounter_date:
        entities.setdefault("dates", [])
        if encounter.encounter_date not in entities["dates"]:
            entities["dates"].append(encounter.encounter_date)
    if image_path:
        entities["image_path"] = image_path

    bboxes = resolve_bounding_boxes(
        raw_text,
        page_start=page_start,
        page_end=page_end,
        pages_by_num=pages_by_num,
    )

    meta = ChunkMetadata(
        document_id=document_id,
        source_document=source_document,
        patient_id=patient_id,
        page_start=page_start,
        page_end=page_end,
        section=heading or section_key,
        encounter_id=encounter.encounter_id if encounter else None,
        encounter_date=encounter.encounter_date if encounter else None,
        encounter_type=encounter.encounter_type if encounter else None,
        provider=encounter.provider if encounter else None,
        facility=encounter.facility if encounter else None,
        char_start=char_start,
        char_end=char_end or len(raw_text),
        bounding_boxes=bboxes,
        chunk_type=chunk_type,
        table_type=table_type,
        parent_chunk_id=(
            f"encounter_{encounter.encounter_id}" if encounter else None
        ),
        entities=entities,
    )
    return Chunk(
        chunk_id=chunk_id,
        raw_text=raw_text,
        retrieval_text=retrieval_text,
        normalized_text=normalized,
        metadata=meta,
    )
