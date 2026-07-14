from __future__ import annotations

from pathlib import Path

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import apply_device
from clinical_retrieval.indexing.bm25_index import BM25Index
from clinical_retrieval.indexing.qdrant_dense import QdrantDenseIndex
from clinical_retrieval.indexing.structured_store import StructuredStore
from clinical_retrieval.indexing.visual_index import VisualIndex
from clinical_retrieval.pipeline import load_chunks
from clinical_retrieval.retrieval.hybrid_retriever import HybridRetriever
from clinical_retrieval.retrieval.model_registry import get_or_create
from clinical_retrieval.retrieval.reranker import Reranker
from clinical_retrieval.retrieval.visual_retriever import VisualRetriever


def _profile_overrides(config: AppConfig) -> None:
    profile = (config.retrieval.profile or "full").lower()
    if profile == "api":
        config.retrieval.enable_visual = False
        config.models.visual_enabled = False
        config.retrieval.rerank_top_k = min(config.retrieval.rerank_top_k, 20)
        config.retrieval.fusion_top_k = min(config.retrieval.fusion_top_k, 40)


def build_retriever(
    config: AppConfig,
    load_visual: bool | None = None,
    lazy_visual: bool | None = None,
    device: str | None = None,
) -> HybridRetriever:
    apply_device(config, device if device is not None else config.models.embedding_device)
    _profile_overrides(config)
    chunks = load_chunks(Path(config.paths.processed_dir) / "chunks.jsonl")
    index_dir = Path(config.paths.index_dir)
    bm25 = BM25Index.load(index_dir / "bm25.pkl")

    device_name = config.models.embedding_device
    dense_key = (
        f"dense:{config.models.embedding_model}:{config.qdrant.dense_collection}:{device_name}"
    )

    def _make_dense():
        d = QdrantDenseIndex(
            model_name=config.models.embedding_model,
            device=device_name,
            batch_size=config.models.embedding_batch_size,
            qdrant_cfg=config.qdrant,
            collection=config.qdrant.dense_collection,
            embedding_dim=config.models.embedding_dim,
        )
        d.load_numpy_cache(index_dir)
        return d

    dense = get_or_create(dense_key, _make_dense)

    structured = None
    posting_index = None
    if Path(config.paths.sqlite_path).exists():
        structured = StructuredStore(config.paths.sqlite_path)
        try:
            posting_index = structured.load_posting_index()
        except Exception:
            posting_index = None

    profile = (config.retrieval.profile or "full").lower()
    use_visual_flag = config.models.visual_enabled if load_visual is None else load_visual
    use_visual_flag = bool(use_visual_flag and config.retrieval.enable_visual and profile != "api")

    if lazy_visual is None:
        lazy_visual = profile == "api"

    visual_ret = None
    visual_loader = None

    def _make_visual_index():
        return VisualIndex(
            model_name=config.models.visual_model,
            device=device_name,
            batch_size=config.models.visual_batch_size,
            qdrant_cfg=config.qdrant,
            collection=config.qdrant.visual_collection,
            mv_dir=config.paths.visual_mv_dir,
        )

    def _load_visual():
        vindex = get_or_create(
            f"visual_index:{config.models.visual_model}:{device_name}",
            _make_visual_index,
        )
        return VisualRetriever(vindex, chunks)

    if use_visual_flag and not lazy_visual:
        try:
            visual_ret = _load_visual()
        except Exception as exc:
            print(f"Visual channel disabled: {exc}")
    elif use_visual_flag and lazy_visual:
        visual_loader = _load_visual

    reranker = None
    if config.models.use_reranker:
        reranker = get_or_create(
            f"rerank:{config.models.reranker_model}:{device_name}",
            lambda: Reranker(
                model_name=config.models.reranker_model,
                device=device_name,
            ),
        )

    retriever = HybridRetriever(
        chunks,
        bm25,
        dense,
        config.retrieval,
        reranker=reranker,
        structured=structured,
        visual=visual_ret,
        posting_index=posting_index,
        lexicon_path=config.structure.lexicon_path,
    )
    if visual_loader is not None:
        retriever.set_visual_loader(visual_loader)
    return retriever
