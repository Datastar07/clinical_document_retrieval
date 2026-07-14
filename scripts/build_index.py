#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import add_device_arg, apply_device
from clinical_retrieval.indexing.bm25_index import BM25Index
from clinical_retrieval.indexing.qdrant_dense import QdrantDenseIndex
from clinical_retrieval.indexing.structured_store import StructuredStore
from clinical_retrieval.indexing.visual_index import VisualIndex
from clinical_retrieval.pipeline import load_chunks, load_encounters

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

    root = Path.cwd()
    config = AppConfig.from_yaml(args.config).resolve(root)
    apply_device(config, args.device)
    print(f"Building indexes on device={config.models.embedding_device}")
    chunks_path = Path(args.chunks) if args.chunks else Path(config.paths.processed_dir) / "chunks.jsonl"
    index_dir = Path(config.paths.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks(chunks_path)
    encounters = load_encounters(Path(config.paths.processed_dir) / "encounters.json")
    print(f"Loaded {len(chunks)} chunks")

    # Channel A: BM25
    bm25 = BM25Index(chunks)
    bm25.save(index_dir / "bm25.pkl")
    print("BM25 index saved")

    # Channel D: SQLite structured
    store = StructuredStore(config.paths.sqlite_path)
    stats = store.rebuild(chunks, encounters)
    store.close()
    print(f"SQLite structured store saved: {stats}")

    # Channel B: Qdrant dense (Qwen3)
    if not args.skip_dense:
        dense = QdrantDenseIndex(
            model_name=config.models.embedding_model,
            device=config.models.embedding_device,
            batch_size=config.models.embedding_batch_size,
            qdrant_cfg=config.qdrant,
            collection=config.qdrant.dense_collection,
            embedding_dim=config.models.embedding_dim,
        )
        dense.build(chunks)
        dense.save_numpy_cache(index_dir)
        print(f"Qdrant dense index saved ({config.models.embedding_model})")
        dense.unload()

    # Channel C: Visual ColQwen
    visual_meta = None
    if config.models.visual_enabled and not args.skip_visual:
        img_dir = Path(config.paths.page_images_dir)
        images = sorted(img_dir.glob("page_*.jpg")) + sorted(img_dir.glob("page_*.png"))
        # Prefer jpg unique pages
        by_page: dict[int, Path] = {}
        for p in images:
            try:
                page_no = int(p.stem.split("_")[1])
            except Exception:
                continue
            by_page.setdefault(page_no, p)
        page_numbers = sorted(by_page)
        image_paths = [by_page[p] for p in page_numbers]
        if image_paths:
            visual = VisualIndex(
                model_name=config.models.visual_model,
                device=config.models.embedding_device,
                batch_size=config.models.visual_batch_size,
                qdrant_cfg=config.qdrant,
                collection=config.qdrant.visual_collection,
                mv_dir=config.paths.visual_mv_dir,
            )
            visual.build_from_images(image_paths, page_numbers)
            visual_meta = {
                "backend": visual.backend,
                "model": visual.model_name,
                "n_pages": len(page_numbers),
            }
            visual.unload()
            print(f"Visual index saved ({visual_meta})")
        else:
            print("No page images found; skipping visual index")

    meta = {
        "n_chunks": len(chunks),
        "embedding_model": config.models.embedding_model,
        "reranker_model": config.models.reranker_model,
        "visual_model": config.models.visual_model,
        "sqlite_path": config.paths.sqlite_path,
        "qdrant_path": config.qdrant.path,
        "dense_collection": config.qdrant.dense_collection,
        "visual_collection": config.qdrant.visual_collection,
        "visual": visual_meta,
        "structured": stats,
    }
    with open(index_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Indexes written to {index_dir}")


if __name__ == "__main__":
    main()
