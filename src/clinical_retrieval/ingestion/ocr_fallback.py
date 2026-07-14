"""OCR fallback for pages with insufficient extractable text."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import fitz

logger = logging.getLogger(__name__)


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def ocr_page_image(
    page: fitz.Page,
    *,
    dpi: int = 200,
    lang: str = "eng",
    tesseract_cmd: str = "tesseract",
) -> str:
    """Render a PDF page to PNG and run Tesseract OCR."""
    if not tesseract_available():
        raise RuntimeError(
            "tesseract is not installed or not on PATH; cannot OCR page "
            f"{page.number + 1}"
        )

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)

    with tempfile.TemporaryDirectory(prefix="clinical_ocr_") as tmp:
        img_path = Path(tmp) / f"page_{page.number + 1}.png"
        out_base = Path(tmp) / "ocr_out"
        pix.save(str(img_path))
        cmd = [
            tesseract_cmd,
            str(img_path),
            str(out_base),
            "-l",
            lang,
            "--psm",
            "6",
        ]
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"OCR timed out on page {page.number + 1}") from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"tesseract failed on page {page.number + 1}: {proc.stderr.strip()}"
            )
        txt_path = Path(str(out_base) + ".txt")
        if not txt_path.exists():
            return ""
        return txt_path.read_text(encoding="utf-8", errors="ignore").strip()


def should_ocr_page(text: str, min_text_chars: int) -> bool:
    cleaned = "".join(text.split())
    return len(cleaned) < min_text_chars
