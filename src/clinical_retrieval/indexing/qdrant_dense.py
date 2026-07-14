from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from clinical_retrieval.config import QdrantConfig
from clinical_retrieval.tf_guards import apply_tf_guards

logger = logging.getLogger(__name__)


def make_qdrant_client(cfg: QdrantConfig) -> QdrantClient:
    if cfg.url:
        return QdrantClient(
            url=cfg.url,
            prefer_grpc=cfg.prefer_grpc,
            check_compatibility=False,
        )
    path = Path(cfg.path)
    path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(path))


class QdrantDenseIndex:
    """Dense chunk index backed by Qdrant + Qwen3 (or other ST models)."""

    def __init__(
        self,
        *,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        batch_size: int = 32,
        qdrant_cfg: QdrantConfig | None = None,
        collection: str | None = None,
        embedding_dim: int = 1024,
    ):
        apply_tf_guards()
        import torch
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.qdrant_cfg = qdrant_cfg or QdrantConfig()
        self.collection = collection or self.qdrant_cfg.dense_collection
        self.client = make_qdrant_client(self.qdrant_cfg)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.chunk_ids: list[str] = []
        # Optional local cache for offline fallback / ablations
        self.embeddings: np.ndarray | None = None
        self._is_qwen = "qwen3" in model_name.lower()

    def _ensure_collection(self, dim: int) -> None:
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection in names:
            return
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def _encode_docs(self, texts: list[str]) -> np.ndarray:
        from tqdm import tqdm

        vectors: list[np.ndarray] = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Embedding chunks"):
            batch = texts[i : i + self.batch_size]
            emb = self.model.encode(
                batch,
                batch_size=self.batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
            vectors.append(emb.astype(np.float32))
        return np.vstack(vectors) if vectors else np.zeros((0, self.embedding_dim), np.float32)

    def _encode_query(self, query: str) -> np.ndarray:
        kwargs: dict[str, Any] = {
            "normalize_embeddings": True,
            "convert_to_numpy": True,
        }
        # Qwen3 embedding uses an instruction prompt for queries
        if self._is_qwen:
            try:
                emb = self.model.encode([query], prompt_name="query", **kwargs)[0]
                return emb.astype(np.float32)
            except Exception:
                pass
        emb = self.model.encode([query], **kwargs)[0]
        return emb.astype(np.float32)

    def build(self, chunks) -> None:
        texts = [c.retrieval_text for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.embeddings = self._encode_docs(texts)
        dim = int(self.embeddings.shape[1]) if self.embeddings.size else self.embedding_dim
        self.embedding_dim = dim

        # Upsert into Qdrant
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection in names:
            self.client.delete_collection(self.collection)
        self._ensure_collection(dim)

        points = []
        for i, c in enumerate(chunks):
            payload = {
                "chunk_id": c.chunk_id,
                "page_start": c.metadata.page_start,
                "page_end": c.metadata.page_end,
                "section": c.metadata.section,
                "encounter_id": c.metadata.encounter_id,
                "encounter_date": c.metadata.encounter_date,
                "chunk_type": c.metadata.chunk_type,
            }
            points.append(
                qm.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
                    vector=self.embeddings[i].tolist(),
                    payload=payload,
                )
            )
            if len(points) >= 256:
                self.client.upsert(collection_name=self.collection, points=points)
                points = []
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
        logger.info("Upserted %d dense vectors into %s", len(chunks), self.collection)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        q = self._encode_query(query)
        try:
            # qdrant-client >=1.12 uses query_points
            if hasattr(self.client, "query_points"):
                resp = self.client.query_points(
                    collection_name=self.collection,
                    query=q.tolist(),
                    limit=top_k,
                    with_payload=True,
                )
                hits = resp.points
            else:
                hits = self.client.search(
                    collection_name=self.collection,
                    query_vector=q.tolist(),
                    limit=top_k,
                    with_payload=True,
                )
            out: list[tuple[str, float]] = []
            for h in hits:
                payload = h.payload or {}
                cid = payload.get("chunk_id")
                if cid:
                    out.append((cid, float(h.score)))
            return out
        except Exception as exc:
            logger.warning("Qdrant dense search failed (%s); using local matmul", exc)
            if self.embeddings is None or not self.chunk_ids:
                return []
            scores = self.embeddings @ q
            idx = np.argsort(scores)[::-1][:top_k]
            return [(self.chunk_ids[i], float(scores[i])) for i in idx]

    def save_numpy_cache(self, index_dir: Path) -> None:
        """Optional numpy cache for ablation / offline."""
        import json

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(index_dir / "dense_embeddings.npy", self.embeddings)
        with open(index_dir / "dense_chunk_ids.json", "w", encoding="utf-8") as f:
            json.dump({"chunk_ids": self.chunk_ids, "model": self.model_name}, f)

    def load_numpy_cache(self, index_dir: Path) -> None:
        import json

        index_dir = Path(index_dir)
        npy = index_dir / "dense_embeddings.npy"
        meta = index_dir / "dense_chunk_ids.json"
        if npy.exists() and meta.exists():
            self.embeddings = np.load(npy)
            with open(meta, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.chunk_ids = data["chunk_ids"]

    def unload(self) -> None:
        """Free GPU memory from embedding model."""
        import gc
        import torch

        del self.model
        self.model = None  # type: ignore
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
