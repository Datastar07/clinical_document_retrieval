from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import fitz

from clinical_retrieval.tf_guards import apply_tf_guards

logger = logging.getLogger(__name__)


def run_docling_extract(
    pdf_path: str | Path,
    out_dir: str | Path,
    *,
    enabled: bool = True,
    mode: str = "batched",
    max_pages_inline: int = 200,
    batch_pages: int = 50,
) -> dict[str, Any] | None:
    """
    Run Docling conversion and write layout artifacts.

    Modes:
      - inline: single convert (skips when pages > max unless FORCE_DOCLING)
      - batched: split large PDFs into page ranges; small PDFs still convert inline
      - skip: write skipped summary
    """
    if not enabled:
        return None

    apply_tf_guards()
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except Exception as exc:
        logger.warning("Docling import failed: %s", exc)
        return None

    force = os.environ.get("FORCE_DOCLING", "").strip().lower() in {"1", "true", "yes"}
    src = fitz.open(pdf_path)
    n_pages = src.page_count

    if mode == "skip" and not force:
        summary = {"status": "skipped", "pages": n_pages, "reason": "docling_mode=skip"}
        (out_dir / "docling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        src.close()
        return summary

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    if hasattr(pipeline_options, "do_table_structure"):
        pipeline_options.do_table_structure = True
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

    try:
        # Small PDFs (or FORCE): always single-shot convert
        if n_pages <= max_pages_inline or force:
            result = converter.convert(str(pdf_path))
            summary = _write_artifacts(result.document, out_dir, pages=n_pages)
            src.close()
            return summary

        if mode != "batched":
            summary = {
                "status": "skipped_large_pdf",
                "pages": n_pages,
                "reason": f"pages>{max_pages_inline}; set FORCE_DOCLING=1 or docling_mode=batched",
            }
            (out_dir / "docling_summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            src.close()
            return summary

        logger.info("Docling batched convert: %d pages, batch=%d", n_pages, batch_pages)
        all_headings: list[dict[str, Any]] = []
        all_tables: list[dict[str, Any]] = []
        md_parts: list[str] = []
        checkpoint = out_dir / "docling_checkpoint.json"
        start_batch = 0
        if checkpoint.exists():
            try:
                start_batch = int(json.loads(checkpoint.read_text()).get("next_batch_start", 0))
            except Exception:
                start_batch = 0

        with tempfile.TemporaryDirectory(prefix="docling_batch_") as tmp:
            tmp_path = Path(tmp)
            for start in range(start_batch, n_pages, batch_pages):
                end = min(n_pages, start + batch_pages)
                batch_pdf = tmp_path / f"batch_{start+1}_{end}.pdf"
                batch_doc = fitz.open()
                batch_doc.insert_pdf(src, from_page=start, to_page=end - 1)
                batch_doc.save(str(batch_pdf))
                batch_doc.close()
                try:
                    result = converter.convert(str(batch_pdf))
                    doc = result.document
                    headings = _collect_headings(doc)
                    for h in headings:
                        if h.get("page") is not None:
                            h["page"] = int(h["page"]) + start
                        else:
                            h["page"] = start + 1
                    tables = _collect_tables(doc)
                    for t in tables:
                        if t.get("page") is not None:
                            t["page"] = int(t["page"]) + start
                        else:
                            t["page"] = start + 1
                    all_headings.extend(headings)
                    all_tables.extend(tables)
                    try:
                        md_parts.append(doc.export_to_markdown())
                    except Exception:
                        pass
                except Exception as exc:
                    logger.warning("Docling batch %s-%s failed: %s", start + 1, end, exc)
                checkpoint.write_text(
                    json.dumps({"next_batch_start": end}, indent=2), encoding="utf-8"
                )
                logger.info("Docling batch done pages %d-%d", start + 1, end)

        src.close()
        (out_dir / "docling_headings.json").write_text(
            json.dumps(all_headings, indent=2), encoding="utf-8"
        )
        (out_dir / "docling_tables.json").write_text(
            json.dumps(all_tables, indent=2), encoding="utf-8"
        )
        (out_dir / "docling_export.md").write_text("\n\n".join(md_parts), encoding="utf-8")
        summary = {
            "status": "ok",
            "mode": "batched",
            "pages": n_pages,
            "n_headings": len(all_headings),
            "n_tables": len(all_tables),
            "markdown_path": str(out_dir / "docling_export.md"),
        }
        (out_dir / "docling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if checkpoint.exists():
            checkpoint.unlink()
        return summary
    except Exception as exc:
        logger.warning("Docling conversion failed: %s", exc)
        try:
            src.close()
        except Exception:
            pass
        summary = {"status": "failed", "error": str(exc)}
        (out_dir / "docling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return None


def _write_artifacts(doc: Any, out_dir: Path, pages: int | None = None) -> dict[str, Any]:
    md_path = out_dir / "docling_export.md"
    json_path = out_dir / "docling_export.json"
    try:
        md_path.write_text(doc.export_to_markdown(), encoding="utf-8")
    except Exception as exc:
        logger.warning("Docling markdown export failed: %s", exc)
    payload: dict[str, Any] = {}
    try:
        if hasattr(doc, "export_to_dict"):
            payload = doc.export_to_dict()
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
    except Exception as exc:
        logger.warning("Docling JSON export failed: %s", exc)

    headings = _collect_headings(doc)
    tables = _collect_tables(doc)
    (out_dir / "docling_headings.json").write_text(json.dumps(headings, indent=2), encoding="utf-8")
    (out_dir / "docling_tables.json").write_text(json.dumps(tables, indent=2), encoding="utf-8")
    summary = {
        "status": "ok",
        "mode": "inline",
        "markdown_path": str(md_path),
        "json_path": str(json_path),
        "n_headings": len(headings),
        "n_tables": len(tables),
        "pages": pages or getattr(doc, "num_pages", None),
    }
    (out_dir / "docling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _collect_headings(doc: Any) -> list[dict[str, Any]]:
    headings: list[dict[str, Any]] = []
    for t in getattr(doc, "texts", None) or []:
        label = str(getattr(t, "label", "") or "").lower()
        if "title" in label or "section" in label or "heading" in label:
            headings.append(
                {"text": getattr(t, "text", "") or "", "label": label, "page": _page_no(t)}
            )
    return headings


def _collect_tables(doc: Any) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for i, table in enumerate(getattr(doc, "tables", None) or []):
        try:
            md = table.export_to_markdown(doc=doc) if hasattr(table, "export_to_markdown") else ""
        except Exception:
            md = ""
        tables.append({"index": i, "page": _page_no(table), "markdown": md[:4000]})
    return tables


def _page_no(item: Any) -> int | None:
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    try:
        first = prov[0]
        page = getattr(first, "page_no", None)
        return int(page) if page is not None else None
    except Exception:
        return None
