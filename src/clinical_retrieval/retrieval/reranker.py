from __future__ import annotations

from clinical_retrieval.tf_guards import apply_tf_guards

apply_tf_guards()

import torch
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self, model_name: str = "Qwen/Qwen3-Reranker-0.6B", device: str = "cuda"):
        use_cuda = torch.cuda.is_available() and device.startswith("cuda")
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device="cuda" if use_cuda else "cpu")

    def score(self, query: str, texts: list[str], batch_size: int = 8) -> list[float]:
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        return [float(s) for s in scores]
