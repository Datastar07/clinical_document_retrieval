from __future__ import annotations

import math
from typing import Any


def hit_at_k(ranks: list[int | None], k: int) -> float:
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr_at_k(ranks: list[int | None], k: int = 10) -> float:
    if not ranks:
        return 0.0
    total = 0.0
    for r in ranks:
        if r is not None and r <= k:
            total += 1.0 / r
    return total / len(ranks)


def ndcg_at_k(ranks: list[int | None], k: int = 10) -> float:
    """Binary relevance nDCG@k using first relevant rank only."""
    if not ranks:
        return 0.0
    scores = []
    for r in ranks:
        dcg = 0.0
        if r is not None and r <= k:
            dcg = 1.0 / math.log2(r + 1)
        idcg = 1.0
        scores.append(dcg / idcg)
    return sum(scores) / len(scores)


def precision_at_k(per_query: list[dict[str, Any]], k: int = 10) -> float:
    """Fraction of top-k slots that are relevant (binary: only first hit known)."""
    if not per_query:
        return 0.0
    vals = []
    for q in per_query:
        r = q.get("first_relevant_rank")
        vals.append(1.0 / k if r is not None and r <= k else 0.0)
    return sum(vals) / len(vals)


def page_hit_at_k(per_query: list[dict[str, Any]], k: int = 10) -> float:
    if not per_query:
        return 0.0
    hits = 0
    for q in per_query:
        expected = set(q.get("expected_pages") or [])
        found = False
        for r in (q.get("top_results") or [])[:k]:
            meta = r.get("metadata") or {}
            pages = set(
                range(meta.get("page_start", 0), meta.get("page_end", 0) + 1)
            )
            if pages & expected:
                found = True
                break
        if found:
            hits += 1
    return hits / len(per_query)


def encounter_hit_at_k(per_query: list[dict[str, Any]], k: int = 10) -> float:
    """Proxy: whether any top-k result shares page with GT (encounter IDs often absent in GT)."""
    return page_hit_at_k(per_query, k=k)


def duplicate_result_ratio(per_query: list[dict[str, Any]]) -> float:
    if not per_query:
        return 0.0
    ratios = []
    for q in per_query:
        texts = [(r.get("content") or "")[:160] for r in (q.get("top_results") or [])]
        if not texts:
            ratios.append(0.0)
            continue
        uniq = len(set(texts))
        ratios.append(1.0 - (uniq / len(texts)))
    return sum(ratios) / len(ratios)


def aggregate_metrics(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    ranks = [q.get("first_relevant_rank") for q in per_query]
    page_acc = [
        1.0 if q.get("page_overlap") else 0.0
        for q in per_query
        if q.get("first_relevant_rank") is not None
    ]
    return {
        "n_queries": len(per_query),
        "hit_at_1": hit_at_k(ranks, 1),
        "hit_at_3": hit_at_k(ranks, 3),
        "hit_at_5": hit_at_k(ranks, 5),
        "hit_at_10": hit_at_k(ranks, 10),
        "recall_at_10": hit_at_k(ranks, 10),
        "mrr_at_10": mrr_at_k(ranks, 10),
        "ndcg_at_10": ndcg_at_k(ranks, 10),
        "precision_at_10": precision_at_k(per_query, 10),
        "page_hit_at_10": page_hit_at_k(per_query, 10),
        "encounter_hit_at_10": encounter_hit_at_k(per_query, 10),
        "duplicate_result_ratio": duplicate_result_ratio(per_query),
        "mean_first_relevant_rank": (
            sum(r for r in ranks if r is not None) / max(1, sum(1 for r in ranks if r is not None))
            if any(r is not None for r in ranks)
            else None
        ),
        "page_overlap_accuracy": sum(page_acc) / len(page_acc) if page_acc else 0.0,
        "missed_queries": [q["query_id"] for q in per_query if not q.get("hit_at_10")],
    }
