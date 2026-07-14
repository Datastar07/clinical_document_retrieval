#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import add_device_arg, apply_device
from clinical_retrieval.indexing.build import run_build_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BM25 + Qdrant dense + SQLite + visual indexes")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--chunks", default=None)
    parser.add_argument("--skip-visual", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    add_device_arg(parser)
    args = parser.parse_args()

    from pathlib import Path

    root = Path.cwd()
    config = AppConfig.from_yaml(args.config).resolve(root)
    apply_device(config, args.device)
    print(f"Building indexes on device={config.models.embedding_device}")
    meta = run_build_index(
        config,
        chunks_path=args.chunks,
        skip_visual=args.skip_visual,
        skip_dense=args.skip_dense,
    )
    print(f"Indexes written to {config.paths.index_dir} ({meta.get('n_chunks')} chunks)")


if __name__ == "__main__":
    main()
