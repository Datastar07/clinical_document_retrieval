#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from clinical_retrieval.config import AppConfig
from clinical_retrieval.pipeline import run_ingest


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest clinical PDF into structured chunks")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default=None, help="Override PDF path")
    parser.add_argument("--output", default=None, help="Override processed output dir")
    args = parser.parse_args()

    root = Path.cwd()
    config = AppConfig.from_yaml(args.config).resolve(root)
    if args.input:
        config.document.source_pdf = str(Path(args.input).resolve())
    if args.output:
        config.paths.processed_dir = str(Path(args.output).resolve())

    summary = run_ingest(config)
    print(f"Ingest complete: {summary['chunks']} chunks from {summary['pages']} pages")
    print(f"Encounters: {summary['encounters']} | Sections: {summary['sections']}")
    print(f"Output: {config.paths.processed_dir}")


if __name__ == "__main__":
    main()
