from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from clinical_retrieval.schemas import Chunk
from clinical_retrieval.structure.normalizer import normalize_text


TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?|[a-z]\d{2}(?:\.\d+)?", re.I)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.corpus_tokens = [tokenize(c.retrieval_text + " " + c.normalized_text) for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        if len(scores) == 0:
            return []
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in idx if scores[i] > 0]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "corpus_tokens": self.corpus_tokens}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.chunk_ids = data["chunk_ids"]
        obj.corpus_tokens = data["corpus_tokens"]
        obj.bm25 = BM25Okapi(obj.corpus_tokens)
        return obj
