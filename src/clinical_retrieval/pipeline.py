from __future__ import annotations

import json
from pathlib import Path

from clinical_retrieval.chunking.chunk_builder import build_chunks
from clinical_retrieval.config import AppConfig
from clinical_retrieval.ingestion.docling_extract import run_docling_extract
from clinical_retrieval.ingestion.page_renderer import render_pdf_pages
from clinical_retrieval.ingestion.pdf_validator import validate_pdf
from clinical_retrieval.ingestion.pymupdf_extractor import extract_pages
from clinical_retrieval.schemas import Chunk, EncounterMeta
from clinical_retrieval.structure.base import get_parser, list_parsers
from clinical_retrieval.structure.lexicon import load_lexicon


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _prepare_parser(config: AppConfig, out_dir: Path):
    """Return a structure parser, wiring processed_dir for auto/docling."""
    name = (config.structure.parser or "auto").strip().lower()
    # Warm lexicon cache for entity extractors / query expansion
    load_lexicon(config.structure.lexicon_path)

    from clinical_retrieval.structure.parsers import AutoStructureParser, DoclingStructureParser

    if name == "auto":
        parser = AutoStructureParser(out_dir)
        return parser
    if name == "docling":
        parser = DoclingStructureParser(out_dir)
        return parser
    try:
        return get_parser(name)
    except Exception:
        return get_parser(config.structure.docling_fallback)


def run_ingest(config: AppConfig) -> dict:
    pdf = Path(config.document.source_pdf)
    out_dir = Path(config.paths.processed_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    page_images_dir = Path(config.paths.page_images_dir)
    page_images_dir.mkdir(parents=True, exist_ok=True)

    report = validate_pdf(
        pdf,
        processing_version=config.document.processing_version,
        min_text_chars=config.extraction.min_text_chars_per_page,
    )
    save_json(report.model_dump(), out_dir / "validation.json")

    # Dual path: Docling layout (best-effort) + PyMuPDF coordinates/text
    run_docling = config.structure.docling_enabled and config.structure.parser in {
        "docling",
        "auto",
    }
    docling_summary = run_docling_extract(
        pdf,
        out_dir,
        enabled=run_docling,
        mode=config.structure.docling_mode,
        max_pages_inline=config.structure.docling_max_pages_inline,
        batch_pages=config.structure.docling_batch_pages,
    )

    pages = extract_pages(
        pdf,
        document_id=config.document.document_id,
        furniture_threshold=config.extraction.header_footer_repeat_threshold,
        ocr_enabled=config.extraction.ocr_enabled,
        min_text_chars=config.extraction.min_text_chars_per_page,
        ocr_dpi=config.extraction.ocr_dpi,
        ocr_lang=config.extraction.ocr_lang,
        ocr_max_pages=config.extraction.ocr_max_pages,
    )
    save_jsonl([p.model_dump() for p in pages], out_dir / "pages.jsonl")

    if config.extraction.render_pages:
        render_pdf_pages(
            pdf,
            page_images_dir,
            dpi=config.extraction.render_dpi,
            fmt=config.extraction.render_format,
            quality=config.extraction.render_quality,
        )

    parser = _prepare_parser(config, out_dir)
    encounters = parser.parse_encounters(pages)
    selected_name = getattr(parser, "selected", None) or parser.name
    if not encounters and selected_name in {"docling", "auto"}:
        fallback = get_parser(config.structure.docling_fallback)
        encounters = fallback.parse_encounters(pages)
        parser = fallback
        selected_name = parser.name
    save_json([e.model_dump() for e in encounters], out_dir / "encounters.json")

    sections = parser.parse_sections(pages, encounters)
    save_json([s.model_dump() for s in sections], out_dir / "sections.json")

    chunks = build_chunks(
        pages=pages,
        sections=sections,
        encounters=encounters,
        document_id=config.document.document_id,
        patient_id=config.document.patient_id,
        patient_name=config.document.patient_name,
        source_document=pdf.name,
        config=config.chunking,
        page_images_dir=page_images_dir if config.chunking.create_page_visual_chunks else None,
    )
    save_jsonl([c.model_dump() for c in chunks], out_dir / "chunks.jsonl")

    ocr_still_needed = [p.page_number for p in pages if p.ocr_required]
    ocr_applied = sum(1 for p in pages if any(b.block_id.endswith("_ocr") for b in p.blocks))
    bbox_filled = sum(1 for c in chunks if c.metadata.bounding_boxes)

    summary = {
        "pages": len(pages),
        "encounters": len(encounters),
        "sections": len(sections),
        "chunks": len(chunks),
        "chunks_with_bbox": bbox_filled,
        "structure_parser": getattr(parser, "selected", None) or parser.name,
        "structure_parser_config": config.structure.parser,
        "available_structure_parsers": list_parsers(),
        "docling": docling_summary,
        "page_images_dir": str(page_images_dir),
        "ocr_enabled": config.extraction.ocr_enabled,
        "ocr_still_needed_pages": ocr_still_needed[:50],
        "ocr_applied_pages_count": ocr_applied,
        "validation": report.model_dump(),
    }
    save_json(summary, out_dir / "ingest_summary.json")
    return summary


def load_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.model_validate(json.loads(line)))
    return chunks


def load_encounters(path: Path) -> list[EncounterMeta]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EncounterMeta.model_validate(x) for x in data]
