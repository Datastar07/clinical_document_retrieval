from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    result_lists: list[list[tuple[str, float]]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Standard or weighted RRF over ranked lists of (id, score)."""
    scores: dict[str, float] = defaultdict(float)
    if weights is None:
        weights = [1.0] * len(result_lists)
    assert len(weights) == len(result_lists)
    for weight, result_list in zip(weights, result_lists):
        for rank, (chunk_id, _score) in enumerate(result_list, start=1):
            scores[chunk_id] += float(weight) * (1.0 / (k + rank))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
