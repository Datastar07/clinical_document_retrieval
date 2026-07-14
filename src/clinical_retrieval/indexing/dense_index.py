from __future__ import annotations

import json
from pathlib import Path

from clinical_retrieval.tf_guards import apply_tf_guards

apply_tf_guards()

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from clinical_retrieval.schemas import Chunk


class DenseIndex:
    """Local numpy dense index (ablation / offline fallback)."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=self.device)
        self.chunk_ids: list[str] = []
        self.embeddings: np.ndarray | None = None
        self._is_qwen = "qwen3" in model_name.lower()

    def build(self, chunks: list[Chunk]) -> None:
        texts = [c.retrieval_text for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]
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
        self.embeddings = (
            np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)
        )

    def encode_query(self, query: str) -> np.ndarray:
        kwargs = dict(normalize_embeddings=True, convert_to_numpy=True)
        if self._is_qwen:
            try:
                return self.model.encode([query], prompt_name="query", **kwargs)[0].astype(
                    np.float32
                )
            except Exception:
                pass
        return self.model.encode([query], **kwargs)[0].astype(np.float32)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        if self.embeddings is None or len(self.chunk_ids) == 0:
            return []
        q = self.encode_query(query)
        scores = self.embeddings @ q
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in idx]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        assert self.embeddings is not None
        np.save(index_dir / "dense_embeddings.npy", self.embeddings)
        with open(index_dir / "dense_chunk_ids.json", "w", encoding="utf-8") as f:
            json.dump({"chunk_ids": self.chunk_ids, "model": self.model_name}, f)

    def load_vectors(self, index_dir: Path) -> None:
        self.embeddings = np.load(index_dir / "dense_embeddings.npy")
        with open(index_dir / "dense_chunk_ids.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        self.chunk_ids = data["chunk_ids"]
