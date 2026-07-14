from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

import fitz

from clinical_retrieval.schemas import ValidationReport


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_pdf(
    pdf_path: str | Path,
    processing_version: str = "1.0.0",
    min_text_chars: int = 40,
) -> ValidationReport:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    digest = sha256_file(path)
    doc = fitz.open(path)
    try:
        page_count = len(doc)
        is_encrypted = bool(doc.is_encrypted)
        ocr_required: list[int] = []
        corrupted: list[int] = []
        text_pages = 0

        for i in range(page_count):
            try:
                text = doc[i].get_text("text") or ""
            except Exception:
                corrupted.append(i + 1)
                ocr_required.append(i + 1)
                continue

            cleaned = re.sub(r"\s+", "", text)
            if len(cleaned) >= min_text_chars:
                text_pages += 1
            else:
                ocr_required.append(i + 1)

        return ValidationReport(
            filename=path.name,
            sha256=digest,
            page_count=page_count,
            is_encrypted=is_encrypted,
            contains_text=text_pages > 0,
            ocr_required_pages=ocr_required,
            processing_version=processing_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            corrupted_pages=corrupted,
        )
    finally:
        doc.close()
