from __future__ import annotations

import logging
from pathlib import Path

import fitz
from tqdm import tqdm

logger = logging.getLogger(__name__)


def render_pdf_pages(
    pdf_path: str | Path,
    out_dir: str | Path,
    *,
    dpi: int = 144,
    fmt: str = "jpeg",
    quality: int = 85,
    max_pages: int | None = None,
) -> list[Path]:
    """Render each PDF page to disk. Returns ordered image paths."""
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    paths: list[Path] = []
    n = doc.page_count if max_pages is None else min(doc.page_count, max_pages)
    for i in tqdm(range(n), desc="Rendering pages"):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        ext = "jpg" if fmt.lower() in {"jpeg", "jpg"} else "png"
        out = out_dir / f"page_{i+1:04d}.{ext}"
        if ext == "jpg":
            pix.save(str(out), jpg_quality=quality)
        else:
            pix.save(str(out))
        paths.append(out)
    doc.close()
    logger.info("Rendered %d pages to %s", len(paths), out_dir)
    return paths


def page_image_path(out_dir: str | Path, page_number: int, fmt: str = "jpeg") -> Path:
    ext = "jpg" if fmt.lower() in {"jpeg", "jpg"} else "png"
    return Path(out_dir) / f"page_{page_number:04d}.{ext}"
