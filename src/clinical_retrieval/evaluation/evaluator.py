from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from clinical_retrieval.evaluation.evidence_matcher import (
    evidence_matches,
    parse_expected_pages,
)
from clinical_retrieval.evaluation.metrics import aggregate_metrics
from clinical_retrieval.retrieval.hybrid_retriever import HybridRetriever
from clinical_retrieval.schemas import EvaluationItem, QueryResult


def load_evaluation(path: str | Path) -> list[EvaluationItem]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [EvaluationItem.model_validate(x) for x in data]


def evaluate_query(
    item: EvaluationItem,
    result: QueryResult,
    *,
    token_threshold: float,
    fuzzy_threshold: float,
) -> dict[str, Any]:
    expected_pages = parse_expected_pages(item.expected_pages)
    first_rank = None
    page_overlap = False
    matched_chunk = None
    for r in result.results:
        pages = set(range(r.metadata.get("page_start", 0), r.metadata.get("page_end", 0) + 1))
        text_hit = evidence_matches(
            r.content,
            item.ground_truth_evidence,
            token_threshold=token_threshold,
            fuzzy_threshold=fuzzy_threshold,
        )
        # Also check retrieval-oriented fields aren't available; content is raw_text
        if text_hit:
            first_rank = r.rank
            page_overlap = bool(pages & expected_pages)
            matched_chunk = r.chunk_id
            break

    return {
        "query_id": item.query_id,
        "query": item.query,
        "category": item.category,
        "expected_pages": sorted(expected_pages),
        "ground_truth_evidence": item.ground_truth_evidence,
        "first_relevant_rank": first_rank,
        "hit_at_10": bool(first_rank is not None and first_rank <= 10),
        "page_overlap": page_overlap,
        "matched_chunk_id": matched_chunk,
        "top_results": [r.model_dump() for r in result.results],
    }


def run_evaluation(
    retriever: HybridRetriever,
    items: list[EvaluationItem],
    *,
    token_threshold: float = 0.55,
    fuzzy_threshold: float = 0.72,
) -> dict[str, Any]:
    per_query: list[dict[str, Any]] = []
    latencies: list[float] = []
    for item in items:
        t0 = time.perf_counter()
        # IMPORTANT: retrieval sees only the query text
        result = retriever.retrieve(item.query, query_id=item.query_id)
        latencies.append(time.perf_counter() - t0)
        detail = evaluate_query(
            item,
            result,
            token_threshold=token_threshold,
            fuzzy_threshold=fuzzy_threshold,
        )
        detail["latency_sec"] = latencies[-1]
        per_query.append(detail)

    summary = aggregate_metrics(per_query)
    if latencies:
        sorted_l = sorted(latencies)
        summary["avg_latency_sec"] = sum(latencies) / len(latencies)
        summary["p95_latency_sec"] = sorted_l[min(len(sorted_l) - 1, int(0.95 * len(sorted_l)))]
    return {"summary": summary, "per_query": per_query}


def write_error_analysis(per_query: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "query_id,first_relevant_rank,hit_at_10,failure_category,notes"
    ]
    for q in per_query:
        if q.get("hit_at_10"):
            cat = "OK"
            notes = f"rank={q.get('first_relevant_rank')}"
        else:
            cat = "RETRIEVAL_MISS"
            top = q.get("top_results") or []
            pages = ",".join(
                str(r["metadata"].get("page_start")) for r in top[:3]
            )
            notes = f"top_pages={pages}"
        lines.append(
            f"{q['query_id']},{q.get('first_relevant_rank') or ''},"
            f"{int(bool(q.get('hit_at_10')))},{cat},{notes}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
