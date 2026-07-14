from __future__ import annotations

from clinical_retrieval.indexing.visual_index import VisualIndex
from clinical_retrieval.schemas import Chunk


class VisualRetriever:
    """Map visual page hits onto text / page_visual chunks."""

    def __init__(self, visual_index: VisualIndex, chunks: list[Chunk]):
        self.visual = visual_index
        self.chunks = chunks
        self.by_page: dict[int, list[Chunk]] = {}
        for c in chunks:
            for p in range(c.metadata.page_start, c.metadata.page_end + 1):
                self.by_page.setdefault(p, []).append(c)

    def search(self, query: str, top_k: int = 30) -> list[tuple[str, float]]:
        page_hits = self.visual.search(query, top_k=top_k)
        scores: dict[str, float] = {}
        for page, score in page_hits:
            page_chunks = self.by_page.get(page, [])
            # Prefer page_visual registry chunk, else section/atomic on page
            preferred = [
                c for c in page_chunks if c.metadata.chunk_type == "page_visual"
            ]
            others = [
                c
                for c in page_chunks
                if c.metadata.chunk_type in {"atomic", "section", "compound", "page"}
            ]
            ordered = preferred + others
            for i, c in enumerate(ordered[:12]):
                # Decay within page so visual contributes page evidence without flooding
                bump = float(score) / (1.0 + 0.15 * i)
                scores[c.chunk_id] = max(scores.get(c.chunk_id, 0.0), bump)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[: top_k * 3]
