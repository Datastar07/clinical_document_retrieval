from __future__ import annotations

import re
from pathlib import Path

from clinical_retrieval.chunking.contextualizer import contextualize_chunk
from clinical_retrieval.config import ChunkingConfig
from clinical_retrieval.schemas import Chunk, EncounterMeta, PageContent, SectionSpan
from clinical_retrieval.structure.table_parser import extract_atomic_facts


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            # Prefer split on paragraph/sentence
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if cut > max_chars * 0.4:
                end = start + cut + 1
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [p for p in parts if p]


def _enc_map(encounters: list[EncounterMeta]) -> dict[str, EncounterMeta]:
    return {e.encounter_id: e for e in encounters}


def build_chunks(
    *,
    pages: list[PageContent],
    sections: list[SectionSpan],
    encounters: list[EncounterMeta],
    document_id: str,
    patient_id: str,
    patient_name: str,
    source_document: str,
    config: ChunkingConfig,
    page_images_dir: str | Path | None = None,
) -> list[Chunk]:
    enc_by_id = _enc_map(encounters)
    pages_by_num = {p.page_number: p for p in pages}
    chunks: list[Chunk] = []
    counter = 0

    def next_id(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{document_id.replace('-', '')}_{prefix}_{counter:05d}"

    def make_chunk(raw_text: str, **kwargs) -> Chunk:
        kwargs.setdefault("pages_by_num", pages_by_num)
        return contextualize_chunk(raw_text, **kwargs)

    # A) Atomic facts from sections
    for sec in sections:
        enc = enc_by_id.get(sec.encounter_id) if sec.encounter_id else None
        # Avoid re-running demographics extractors on every clinical section
        if sec.encounter_id and sec.section in {
            "subjective",
            "objective",
            "plan",
            "assessment",
            "laboratory_results",
            "laboratory_orders",
            "imaging_orders",
            "referrals",
            "clinical_notes",
            "patient_education",
            "follow_up",
            "signature",
            "vitals",
        }:
            facts = extract_atomic_facts(sec.section, sec.text)
        elif not sec.encounter_id:
            facts = extract_atomic_facts(sec.section, sec.text)
        else:
            facts = []

        for fact in facts:
            # Skip low-value note sentences to control corpus size
            if fact.get("table_type") == "note_sentence":
                continue
            search = fact["search_text"]
            chunks.append(
                make_chunk(
                    search if len(search) >= len(fact["raw_text"]) else fact["raw_text"],
                    patient_name=patient_name,
                    patient_id=patient_id,
                    document_id=document_id,
                    source_document=source_document,
                    encounter=enc,
                    section=sec,
                    page_start=sec.start_page,
                    page_end=sec.end_page,
                    chunk_id=next_id("ATOMIC"),
                    chunk_type="atomic",
                    table_type=fact.get("table_type"),
                    char_start=0,
                    char_end=len(fact["raw_text"]),
                )
            )

    # Combined surgical-history chunk from preface atomics
    surg = [
        c
        for c in chunks
        if c.metadata.table_type == "surgical_history" and c.metadata.page_start <= 3
    ]
    if len(surg) >= 2:
        joined = "; ".join(c.raw_text for c in surg)
        syn = SectionSpan(
            section="surgical_history",
            section_heading="Surgical History",
            encounter_id=None,
            start_page=2,
            end_page=2,
            text=joined,
        )
        chunks.append(
            make_chunk(
                joined,
                patient_name=patient_name,
                patient_id=patient_id,
                document_id=document_id,
                source_document=source_document,
                encounter=None,
                section=syn,
                page_start=2,
                page_end=2,
                chunk_id=next_id("ATOMIC"),
                chunk_type="atomic",
                table_type="surgical_history",
            )
        )

    # B) Section chunks
    for sec in sections:
        enc = enc_by_id.get(sec.encounter_id) if sec.encounter_id else None
        pieces = _split_long_text(
            sec.text, config.section_max_chars, config.section_overlap_chars
        )
        for piece in pieces:
            chunks.append(
                make_chunk(
                    piece,
                    patient_name=patient_name,
                    patient_id=patient_id,
                    document_id=document_id,
                    source_document=source_document,
                    encounter=enc,
                    section=sec,
                    page_start=sec.start_page,
                    page_end=sec.end_page,
                    chunk_id=next_id("SECTION"),
                    chunk_type="section",
                    char_start=0,
                    char_end=len(piece),
                )
            )

    # C) Encounter parent / key-facts + multi-section compound chunks
    if config.create_key_facts:
        priority_keys = (
            "vitals",
            "laboratory_results",
            "assessment",
            "plan",
            "imaging_orders",
            "laboratory_orders",
            "referrals",
            "follow_up",
            "signature",
            "clinical_notes",
            "subjective",
            "objective",
        )
        for enc in encounters:
            enc_secs = [s for s in sections if s.encounter_id == enc.encounter_id]
            by_key: dict[str, SectionSpan] = {}
            for s in enc_secs:
                # Prefer longer text if duplicate keys
                if s.section not in by_key or len(s.text) > len(by_key[s.section].text):
                    by_key[s.section] = s

            header = [
                f"Encounter {enc.encounter_id}",
                f"Type: {enc.encounter_type or 'Unknown'}",
                f"Date: {enc.encounter_date or 'Unknown'}",
                f"Provider: {enc.provider or 'Unknown'}",
            ]

            def _clip(text: str, limit: int) -> str:
                text = text.strip()
                return text if len(text) <= limit else text[:limit] + "…"

            # Resolve sections robustly (labeling can be noisy)
            def _find_sec(*keys: str, startswith: str | None = None) -> SectionSpan | None:
                for k in keys:
                    if k in by_key:
                        return by_key[k]
                if startswith:
                    for s in enc_secs:
                        if s.text.lstrip().startswith(startswith):
                            return s
                return None

            lab = _find_sec("laboratory_results")
            assess = _find_sec("assessment")
            plan = _find_sec("plan", startswith="P — Plan") or _find_sec(
                "plan", startswith="P - Plan"
            )
            # If plan text lives under a wrong key, recover by content
            if plan is None:
                for s in enc_secs:
                    if "SGLT2" in s.text or "P — Plan" in s.text[:40] or "P - Plan" in s.text[:40]:
                        if "Rationale:" in s.text or s.text.lstrip().startswith("P"):
                            plan = s
                            break

            compound_parts = []
            for sec_obj, heading, limit in (
                (lab, "Laboratory Results", 700),
                (assess, "A — Assessment / Diagnoses", 700),
                (plan, "P — Plan", 1600),
            ):
                if sec_obj is not None:
                    compound_parts.append(
                        f"[{heading}]\n{_clip(sec_obj.text, limit)}"
                    )
            if len(compound_parts) >= 2:
                compound = "\n\n".join(header + compound_parts)
                page_nums = []
                for sec_obj in (lab, assess, plan):
                    if sec_obj is not None:
                        page_nums.extend([sec_obj.start_page, sec_obj.end_page])
                syn = SectionSpan(
                    section="labs_assessment_plan",
                    section_heading="Labs / Assessment / Plan",
                    encounter_id=enc.encounter_id,
                    start_page=min(page_nums),
                    end_page=max(page_nums),
                    text=compound,
                )
                chunks.append(
                    make_chunk(
                        compound,
                        patient_name=patient_name,
                        patient_id=patient_id,
                        document_id=document_id,
                        source_document=source_document,
                        encounter=enc,
                        section=syn,
                        page_start=syn.start_page,
                        page_end=syn.end_page,
                        chunk_id=next_id("COMPOUND"),
                        chunk_type="compound",
                    )
                )

            # Imaging + follow-up compound (multi-span imaging queries)
            img_fu = []
            for key, limit in (("imaging_orders", 900), ("follow_up", 700), ("plan", 500)):
                if key in by_key:
                    img_fu.append(
                        f"[{by_key[key].section_heading}]\n{_clip(by_key[key].text, limit)}"
                    )
            if "imaging_orders" in by_key and (
                "follow_up" in by_key or "plan" in by_key
            ):
                compound = "\n\n".join(header + img_fu)
                page_nums = [
                    by_key["imaging_orders"].start_page,
                    by_key["imaging_orders"].end_page,
                ]
                for k in ("follow_up", "plan"):
                    if k in by_key:
                        page_nums.extend([by_key[k].start_page, by_key[k].end_page])
                syn = SectionSpan(
                    section="imaging_followup",
                    section_heading="Imaging / Follow-Up",
                    encounter_id=enc.encounter_id,
                    start_page=min(page_nums),
                    end_page=max(page_nums),
                    text=compound,
                )
                chunks.append(
                    make_chunk(
                        compound,
                        patient_name=patient_name,
                        patient_id=patient_id,
                        document_id=document_id,
                        source_document=source_document,
                        encounter=enc,
                        section=syn,
                        page_start=syn.start_page,
                        page_end=syn.end_page,
                        chunk_id=next_id("COMPOUND"),
                        chunk_type="compound",
                    )
                )

            # Plan + lab orders (secondary HTN workups, etc.)
            if "plan" in by_key and "laboratory_orders" in by_key:
                compound = "\n\n".join(
                    header
                    + [
                        f"[P — Plan]\n{_clip(by_key['plan'].text, 1400)}",
                        f"[Laboratory Orders]\n{_clip(by_key['laboratory_orders'].text, 900)}",
                    ]
                )
                page_nums = [
                    by_key["plan"].start_page,
                    by_key["plan"].end_page,
                    by_key["laboratory_orders"].start_page,
                    by_key["laboratory_orders"].end_page,
                ]
                syn = SectionSpan(
                    section="plan_lab_orders",
                    section_heading="Plan / Laboratory Orders",
                    encounter_id=enc.encounter_id,
                    start_page=min(page_nums),
                    end_page=max(page_nums),
                    text=compound,
                )
                chunks.append(
                    make_chunk(
                        compound,
                        patient_name=patient_name,
                        patient_id=patient_id,
                        document_id=document_id,
                        source_document=source_document,
                        encounter=enc,
                        section=syn,
                        page_start=syn.start_page,
                        page_end=syn.end_page,
                        chunk_id=next_id("COMPOUND"),
                        chunk_type="compound",
                    )
                )

            # Compact encounter key-facts: priority sections first, skip long HPI
            parts = list(header)
            budgets = {
                "vitals": 400,
                "laboratory_results": 600,
                "assessment": 700,
                "plan": 1100,
                "imaging_orders": 700,
                "laboratory_orders": 600,
                "referrals": 400,
                "follow_up": 500,
                "signature": 200,
                "clinical_notes": 350,
                "subjective": 250,
                "objective": 250,
            }
            for key in priority_keys:
                if key not in by_key:
                    continue
                parts.append(
                    f"[{by_key[key].section_heading}]\n{_clip(by_key[key].text, budgets.get(key, 500))}"
                )

            key_text = "\n\n".join(parts)
            if len(key_text) > config.encounter_summary_max_chars:
                key_text = key_text[: config.encounter_summary_max_chars]
            syn = SectionSpan(
                section="encounter_key_facts",
                section_heading="Encounter Key Facts",
                encounter_id=enc.encounter_id,
                start_page=enc.start_page,
                end_page=enc.end_page,
                text=key_text,
            )
            chunks.append(
                make_chunk(
                    key_text,
                    patient_name=patient_name,
                    patient_id=patient_id,
                    document_id=document_id,
                    source_document=source_document,
                    encounter=enc,
                    section=syn,
                    page_start=enc.start_page,
                    page_end=enc.end_page,
                    chunk_id=next_id("ENCKEY"),
                    chunk_type="encounter_key_facts",
                )
            )


    # D) Page-level fallback chunks for pages with little structure coverage
    covered_pages = set()
    for c in chunks:
        for p in range(c.metadata.page_start, c.metadata.page_end + 1):
            covered_pages.add(p)
    for page in pages:
        if page.page_number in covered_pages:
            continue
        if len(page.clean_text) < 80:
            continue
        enc = None
        for e in encounters:
            if e.start_page <= page.page_number <= e.end_page:
                enc = e
                break
        syn = SectionSpan(
            section="page",
            section_heading="Page Content",
            encounter_id=enc.encounter_id if enc else None,
            start_page=page.page_number,
            end_page=page.page_number,
            text=page.clean_text,
        )
        for piece in _split_long_text(
            page.clean_text, config.section_max_chars, config.section_overlap_chars
        ):
            chunks.append(
                make_chunk(
                    piece,
                    patient_name=patient_name,
                    patient_id=patient_id,
                    document_id=document_id,
                    source_document=source_document,
                    encounter=enc,
                    section=syn,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    chunk_id=next_id("PAGE"),
                    chunk_type="page",
                )
            )

    # E) Page-visual registry chunks (link page → image path for multimodal retrieval)
    if getattr(config, "create_page_visual_chunks", False) and page_images_dir:
        img_dir = Path(page_images_dir)
        for page in pages:
            enc = None
            for e in encounters:
                if e.start_page <= page.page_number <= e.end_page:
                    enc = e
                    break
            jpg = img_dir / f"page_{page.page_number:04d}.jpg"
            png = img_dir / f"page_{page.page_number:04d}.png"
            image_path = str(jpg if jpg.exists() else png if png.exists() else jpg)
            syn = SectionSpan(
                section="page_visual",
                section_heading="Page Visual",
                encounter_id=enc.encounter_id if enc else None,
                start_page=page.page_number,
                end_page=page.page_number,
                text=page.clean_text[:500],
            )
            chunks.append(
                make_chunk(
                    f"[Page image {page.page_number}]\n{page.clean_text[:800]}",
                    patient_name=patient_name,
                    patient_id=patient_id,
                    document_id=document_id,
                    source_document=source_document,
                    encounter=enc,
                    section=syn,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    chunk_id=next_id("VIS"),
                    chunk_type="page_visual",
                    table_type="page_image",
                    image_path=image_path,
                )
            )

    # Neighbor links
    for i, c in enumerate(chunks):
        if i > 0:
            c.metadata.previous_chunk_id = chunks[i - 1].chunk_id
        if i + 1 < len(chunks):
            c.metadata.next_chunk_id = chunks[i + 1].chunk_id

    # Deduplicate identical retrieval texts (keep first)
    seen: set[str] = set()
    unique: list[Chunk] = []
    for c in chunks:
        key = (c.retrieval_text[:500], c.metadata.page_start, c.metadata.chunk_type)
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)
    return unique
