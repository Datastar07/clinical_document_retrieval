from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from qdrant_client.http import models as qm
from tqdm import tqdm

from clinical_retrieval.config import QdrantConfig
from clinical_retrieval.indexing.qdrant_dense import make_qdrant_client
from clinical_retrieval.tf_guards import apply_tf_guards

logger = logging.getLogger(__name__)


def _maxsim(query_vecs: np.ndarray, doc_vecs: np.ndarray) -> float:
    """ColBERT-style MaxSim: sum over query tokens of max similarity to doc tokens."""
    if query_vecs.size == 0 or doc_vecs.size == 0:
        return 0.0
    # (Q, D)
    sims = query_vecs @ doc_vecs.T
    return float(sims.max(axis=1).sum())


class VisualIndex:
    """
    ColQwen2.5 page visual index.

    Stores mean-pooled centroids in Qdrant for coarse ANN and multi-vector
    shards on disk for MaxSim refinement.
    """

    def __init__(
        self,
        *,
        model_name: str = "vidore/colqwen2.5-v0.2",
        device: str = "cuda",
        batch_size: int = 1,
        qdrant_cfg: QdrantConfig | None = None,
        collection: str | None = None,
        mv_dir: str | Path = "data/indexes/visual_mv",
    ):
        apply_tf_guards()
        import torch

        self.model_name = model_name
        self.device = (
            device if torch.cuda.is_available() and device.startswith("cuda") else "cpu"
        )
        self.batch_size = batch_size
        self.qdrant_cfg = qdrant_cfg or QdrantConfig()
        self.collection = collection or self.qdrant_cfg.visual_collection
        self.mv_dir = Path(mv_dir)
        self.mv_dir.mkdir(parents=True, exist_ok=True)
        self.client = make_qdrant_client(self.qdrant_cfg)
        self.model = None
        self.processor = None
        self.backend = "none"
        self._open_clip_tokenizer = None
        self.page_ids: list[int] = []
        self.centroids: np.ndarray | None = None
        self._load_model()

    def _clear_cuda(self) -> None:
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _load_model(self) -> None:
        import torch

        # Prefer colpali-engine (processor first to fail fast on missing files)
        try:
            from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

            self.processor = ColQwen2_5_Processor.from_pretrained(self.model_name)
            self.model = ColQwen2_5.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            ).eval()
            self.backend = "colpali"
            logger.info("Loaded ColQwen via colpali-engine: %s", self.model_name)
            return
        except Exception as exc:
            logger.warning("colpali-engine load failed: %s", exc)
            self.model = None
            self.processor = None
            self._clear_cuda()

        # Transformers ColQwen2 / ColQwen2.5 path
        try:
            from transformers import ColQwen2ForRetrieval, ColQwen2Processor

            self.processor = ColQwen2Processor.from_pretrained(self.model_name)
            self.model = ColQwen2ForRetrieval.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
            ).eval()
            self.backend = "transformers"
            logger.info("Loaded ColQwen via transformers: %s", self.model_name)
            return
        except Exception as exc:
            logger.warning("transformers ColQwen load failed: %s", exc)
            self.model = None
            self.processor = None
            self._clear_cuda()

        # Compact fallback: OpenCLIP / CLIP page embeddings
        try:
            import open_clip
            import torch

            model, _, preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            model = model.to(self.device).eval()
            self.model = model
            self.processor = preprocess
            self.backend = "open_clip"
            self.model_name = "open_clip/ViT-B-32"
            self._open_clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
            logger.warning("Using OpenCLIP page embedding fallback: ViT-B-32")
            return
        except Exception as exc:
            logger.warning("OpenCLIP fallback failed: %s", exc)
            self._clear_cuda()

        try:
            from sentence_transformers import SentenceTransformer

            fallback = "sentence-transformers/clip-ViT-B-32"
            self.model = SentenceTransformer(fallback, device=self.device)
            self.processor = None
            self.backend = "clip"
            self.model_name = fallback
            logger.warning("Using CLIP page embedding fallback: %s", fallback)
        except Exception as exc:
            raise RuntimeError(f"No visual backend available: {exc}") from exc

    def _ensure_collection(self, dim: int) -> None:
        names = [c.name for c in self.client.get_collections().collections]
        if self.collection in names:
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def _encode_image_multivec(self, image_path: Path) -> np.ndarray:
        """Return (N, D) float32 L2-normalized multi-vector."""
        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        if self.backend == "clip":
            emb = self.model.encode(
                [image], normalize_embeddings=True, convert_to_numpy=True
            )[0].astype(np.float32)
            return emb.reshape(1, -1)

        if self.backend == "open_clip":
            with torch.no_grad():
                tensor = self.processor(image).unsqueeze(0).to(self.device)
                emb = self.model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb.detach().float().cpu().numpy().astype(np.float32)

        with torch.no_grad():
            if self.backend == "colpali":
                batch = self.processor.process_images([image]).to(self.model.device)
                out = self.model(**batch)
            else:
                batch = self.processor(images=[image], return_tensors="pt")
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                out = self.model(**batch)
                if hasattr(out, "embeddings"):
                    out = out.embeddings
            if isinstance(out, tuple):
                out = out[0]
            vec = out[0].detach().float().cpu().numpy()
        # L2 normalize tokens
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
        return (vec / norms).astype(np.float32)

    def _encode_query_multivec(self, query: str) -> np.ndarray:
        import torch

        if self.backend == "clip":
            emb = self.model.encode(
                [query], normalize_embeddings=True, convert_to_numpy=True
            )[0].astype(np.float32)
            return emb.reshape(1, -1)

        if self.backend == "open_clip":
            with torch.no_grad():
                tokens = self._open_clip_tokenizer([query]).to(self.device)
                emb = self.model.encode_text(tokens)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                return emb.detach().float().cpu().numpy().astype(np.float32)

        with torch.no_grad():
            if self.backend == "colpali":
                batch = self.processor.process_queries([query]).to(self.model.device)
                out = self.model(**batch)
            else:
                batch = self.processor(text=[query], return_tensors="pt")
                batch = {k: v.to(self.model.device) for k, v in batch.items()}
                out = self.model(**batch)
                if hasattr(out, "embeddings"):
                    out = out.embeddings
            if isinstance(out, tuple):
                out = out[0]
            vec = out[0].detach().float().cpu().numpy()
        norms = np.linalg.norm(vec, axis=1, keepdims=True) + 1e-8
        return (vec / norms).astype(np.float32)

    def build_from_images(self, image_paths: list[Path], page_numbers: list[int]) -> None:
        assert len(image_paths) == len(page_numbers)
        centroids: list[np.ndarray] = []
        self.page_ids = list(page_numbers)
        dim = None
        for path, page in tqdm(
            list(zip(image_paths, page_numbers)), desc="Visual encode pages"
        ):
            mv = self._encode_image_multivec(Path(path))
            np.savez_compressed(self.mv_dir / f"page_{page:04d}.npz", vectors=mv)
            centroid = mv.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids.append(centroid.astype(np.float32))
            dim = int(centroid.shape[0])

        self.centroids = np.vstack(centroids) if centroids else None
        if dim is None:
            return
        self._ensure_collection(dim)
        points = []
        for i, page in enumerate(page_numbers):
            points.append(
                qm.PointStruct(
                    id=page,
                    vector=centroids[i].tolist(),
                    payload={"page": page, "image": str(image_paths[i])},
                )
            )
            if len(points) >= 128:
                self.client.upsert(collection_name=self.collection, points=points)
                points = []
        if points:
            self.client.upsert(collection_name=self.collection, points=points)

        meta = {
            "model": self.model_name,
            "backend": self.backend,
            "n_pages": len(page_numbers),
            "dim": dim,
            "page_ids": page_numbers,
        }
        with open(self.mv_dir / "visual_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def coarse_search(self, query: str, top_k: int = 80) -> list[tuple[int, float]]:
        q_mv = self._encode_query_multivec(query)
        q = q_mv.mean(axis=0)
        q = q / (np.linalg.norm(q) + 1e-8)
        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(
                collection_name=self.collection,
                query=q.astype(np.float32).tolist(),
                limit=top_k,
                with_payload=True,
            )
            hits = resp.points
        else:
            hits = self.client.search(
                collection_name=self.collection,
                query_vector=q.astype(np.float32).tolist(),
                limit=top_k,
                with_payload=True,
            )
        return [(int(h.payload["page"]), float(h.score)) for h in hits if h.payload]

    def maxsim_rerank(
        self, query: str, candidate_pages: list[int]
    ) -> list[tuple[int, float]]:
        q_mv = self._encode_query_multivec(query)
        scored: list[tuple[int, float]] = []
        for page in candidate_pages:
            path = self.mv_dir / f"page_{page:04d}.npz"
            if not path.exists():
                continue
            doc_mv = np.load(path)["vectors"].astype(np.float32)
            scored.append((page, _maxsim(q_mv, doc_mv)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def search(self, query: str, top_k: int = 30, coarse_k: int = 80) -> list[tuple[int, float]]:
        coarse = self.coarse_search(query, top_k=coarse_k)
        pages = [p for p, _ in coarse]
        if self.backend in {"clip", "open_clip"}:
            return coarse[:top_k]
        refined = self.maxsim_rerank(query, pages)
        return refined[:top_k] if refined else coarse[:top_k]

    def unload(self) -> None:
        import torch

        del self.model
        del self.processor
        self.model = None
        self.processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
