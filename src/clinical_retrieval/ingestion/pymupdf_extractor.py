from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path

import fitz

from clinical_retrieval.ingestion.ocr_fallback import ocr_page_image, should_ocr_page
from clinical_retrieval.schemas import PageContent, TextBlock

logger = logging.getLogger(__name__)


HEADER_FOOTER_PATTERNS = [
    re.compile(r"^Patient:\s*.+\|\s*ID:\s*PT-\d+", re.I),
    re.compile(r"^SYNTHETIC DATA", re.I),
    re.compile(r"^CONFIDENTIAL", re.I),
    re.compile(r"^Generated:\s*\d{4}-\d{2}-\d{2}", re.I),
    re.compile(r"^Page\s+\d+\s*$", re.I),
]


def _normalize_furniture(line: str) -> str:
    line = re.sub(r"\d+", "#", line)
    line = re.sub(r"\s+", " ", line).strip().lower()
    return line


def detect_repeated_furniture(pages_text: list[str], threshold: float = 0.6) -> set[str]:
    """Detect highly repeated top/bottom lines across pages."""
    counter: Counter[str] = Counter()
    n = max(len(pages_text), 1)
    for text in pages_text:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        sample = lines[:6] + lines[-4:]
        for ln in sample:
            counter[_normalize_furniture(ln)] += 1
    return {k for k, v in counter.items() if v / n >= threshold and k}


def is_furniture_line(line: str, repeated: set[str]) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    for pat in HEADER_FOOTER_PATTERNS:
        if pat.search(stripped):
            return True
    return _normalize_furniture(stripped) in repeated


def clean_page_text(text: str, repeated: set[str]) -> str:
    kept: list[str] = []
    for line in text.splitlines():
        if is_furniture_line(line, repeated):
            continue
        kept.append(line)
    out: list[str] = []
    blank = 0
    for ln in kept:
        if not ln.strip():
            blank += 1
            if blank <= 1:
                out.append("")
        else:
            blank = 0
            out.append(ln.rstrip())
    return "\n".join(out).strip()


def _extract_blocks(page: fitz.Page, page_number: int) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    raw_dict = page.get_text("dict")
    order = 0
    for bi, block in enumerate(raw_dict.get("blocks", [])):
        if block.get("type") != 0:
            continue
        lines_text: list[str] = []
        sizes: list[float] = []
        bold_flags: list[bool] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_parts = []
            for span in spans:
                t = span.get("text", "")
                if t:
                    line_parts.append(t)
                    sizes.append(float(span.get("size", 0)))
                    bold_flags.append(bool(span.get("flags", 0) & 2**4))
            if line_parts:
                lines_text.append("".join(line_parts))
        block_text = "\n".join(lines_text).strip()
        if not block_text:
            continue
        bbox = [float(x) for x in block.get("bbox", [0, 0, 0, 0])]
        avg_size = sum(sizes) / len(sizes) if sizes else None
        blocks.append(
            TextBlock(
                block_id=f"p{page_number}_e{bi}",
                text=block_text,
                bbox=bbox,
                reading_order=order,
                font_size=avg_size,
                is_bold=any(bold_flags),
                block_type="text",
            )
        )
        order += 1
    return blocks


def extract_pages(
    pdf_path: str | Path,
    document_id: str,
    furniture_threshold: float = 0.6,
    *,
    ocr_enabled: bool = False,
    min_text_chars: int = 40,
    ocr_dpi: int = 200,
    ocr_lang: str = "eng",
    ocr_max_pages: int | None = None,
) -> list[PageContent]:
    """Extract pages; optionally OCR only low-text pages when enabled."""
    path = Path(pdf_path)
    doc = fitz.open(path)
    ocr_used_pages: list[int] = []
    try:
        raw_texts: list[str] = []
        page_meta: list[tuple[int, float, float, str, list[TextBlock], bool]] = []

        for i in range(len(doc)):
            page = doc[i]
            width, height = float(page.rect.width), float(page.rect.height)
            text = page.get_text("text") or ""
            used_ocr = False

            if ocr_enabled and should_ocr_page(text, min_text_chars):
                if ocr_max_pages is not None and len(ocr_used_pages) >= ocr_max_pages:
                    logger.warning(
                        "OCR page budget reached (%s); skipping OCR for page %s",
                        ocr_max_pages,
                        i + 1,
                    )
                else:
                    try:
                        ocr_text = ocr_page_image(page, dpi=ocr_dpi, lang=ocr_lang)
                        if len(re.sub(r"\s+", "", ocr_text)) > len(re.sub(r"\s+", "", text)):
                            text = ocr_text
                            used_ocr = True
                            ocr_used_pages.append(i + 1)
                            logger.info("OCR applied to page %s (%s chars)", i + 1, len(ocr_text))
                    except Exception as exc:
                        logger.warning("OCR failed on page %s: %s", i + 1, exc)

            raw_texts.append(text)
            blocks = _extract_blocks(page, i + 1)
            if used_ocr and text.strip():
                # Add a synthetic OCR block spanning the page for grounding
                blocks.append(
                    TextBlock(
                        block_id=f"p{i+1}_ocr",
                        text=text,
                        bbox=[0.0, 0.0, width, height],
                        reading_order=len(blocks),
                        font_size=None,
                        is_bold=False,
                        block_type="text",
                    )
                )
            page_meta.append((i + 1, width, height, text, blocks, used_ocr))

        repeated = detect_repeated_furniture(raw_texts, threshold=furniture_threshold)
        pages: list[PageContent] = []
        for page_number, width, height, text, blocks, used_ocr in page_meta:
            clean = clean_page_text(text, repeated)
            for b in blocks:
                if is_furniture_line(b.text.splitlines()[0] if b.text else "", repeated):
                    b.block_type = "header" if (b.bbox[1] < height * 0.15) else "footer"
            char_count = len(re.sub(r"\s+", "", clean))
            pages.append(
                PageContent(
                    document_id=document_id,
                    page_number=page_number,
                    width=width,
                    height=height,
                    text=text,
                    clean_text=clean,
                    blocks=blocks,
                    char_count=char_count,
                    ocr_required=(not used_ocr) and char_count < min_text_chars,
                )
            )

        if ocr_used_pages:
            logger.info("OCR completed for %s page(s): %s", len(ocr_used_pages), ocr_used_pages[:20])
        return pages
    finally:
        doc.close()
