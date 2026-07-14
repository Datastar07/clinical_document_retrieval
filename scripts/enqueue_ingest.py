#!/usr/bin/env python3
"""Enqueue clinical PDF ingest jobs (file-backed queue demo)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_retrieval.ingestion.job_queue import FileJobQueue


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, help="Path to clinical PDF")
    ap.add_argument("--document-id", required=True)
    ap.add_argument("--queue-dir", default="data/job_queue")
    ap.add_argument("--processing-version", default="2.0.0")
    args = ap.parse_args()

    pdf = Path(args.pdf).resolve()
    if not pdf.exists():
        raise SystemExit(f"PDF not found: {pdf}")

    q = FileJobQueue(args.queue_dir)
    job = q.enqueue(
        document_id=args.document_id,
        source_pdf=str(pdf),
        processing_version=args.processing_version,
    )
    print(json.dumps({"enqueued": job.to_dict(), "stats": q.stats()}, indent=2))


if __name__ == "__main__":
    main()
