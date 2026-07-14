"""Programmatic index build used by CLI and API upload pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from clinical_retrieval.config import AppConfig
from clinical_retrieval.indexing.bm25_index import BM25Index
from clinical_retrieval.indexing.qdrant_dense import QdrantDenseIndex
from clinical_retrieval.indexing.structured_store import StructuredStore
from clinical_retrieval.indexing.visual_index import VisualIndex
from clinical_retrieval.pipeline import load_chunks, load_encounters

logger = logging.getLogger(__name__)


def run_build_index(
    config: AppConfig,
    *,
    chunks_path: Path | str | None = None,
    skip_visual: bool = False,
    skip_dense: bool = False,
    progress_cb=None,
) -> dict[str, Any]:
    """Rebuild BM25 + SQLite + dense (+ optional visual) from processed chunks."""

    def _progress(stage: str, detail: str | None = None) -> None:
        if progress_cb:
            progress_cb(stage, detail)
        logger.info("%s%s", stage, f" — {detail}" if detail else "")

    chunks_path = Path(chunks_path) if chunks_path else Path(config.paths.processed_dir) / "chunks.jsonl"
    index_dir = Path(config.paths.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    _progress("loading_chunks")
    chunks = load_chunks(chunks_path)
    encounters = load_encounters(Path(config.paths.processed_dir) / "encounters.json")

    _progress("bm25", f"{len(chunks)} chunks")
    bm25 = BM25Index(chunks)
    bm25.save(index_dir / "bm25.pkl")

    _progress("sqlite")
    store = StructuredStore(config.paths.sqlite_path)
    stats = store.rebuild(chunks, encounters)
    store.close()

    if not skip_dense:
        _progress("dense", config.models.embedding_model)
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
        dense.unload()

    visual_meta = None
    if config.models.visual_enabled and not skip_visual:
        img_dir = Path(config.paths.page_images_dir)
        images = sorted(img_dir.glob("page_*.jpg")) + sorted(img_dir.glob("page_*.png"))
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
            _progress("visual", f"{len(image_paths)} pages")
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
        else:
            _progress("visual_skip", "no page images")

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
        "skip_visual": skip_visual,
        "skip_dense": skip_dense,
    }
    with open(index_dir / "index_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _progress("done", str(index_dir))
    return meta
