from __future__ import annotations

"""Grounding quality report for Top-K retrieval results."""

from typing import Any


def build_grounding_report(per_query: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Audit location metadata completeness and page agreement with expected_pages
    when a text hit exists.
    """
    rows: list[dict[str, Any]] = []
    n_hits = 0
    n_page_ok = 0
    n_complete = 0
    field_missing = {
        "document_id": 0,
        "page": 0,
        "section": 0,
        "character_span": 0,
        "bounding_box": 0,
    }

    for q in per_query:
        expected = set(q.get("expected_pages") or [])
        hit = bool(q.get("hit_at_10"))
        matched_id = q.get("matched_chunk_id")
        matched = None
        for r in q.get("top_results") or []:
            if r.get("chunk_id") == matched_id:
                matched = r
                break

        grounding = {
            "query_id": q.get("query_id"),
            "hit_at_10": hit,
            "page_overlap": bool(q.get("page_overlap")),
            "matched_chunk_id": matched_id,
            "fields": {},
            "issues": [],
        }

        if matched:
            n_hits += 1
            meta = matched.get("metadata") or {}
            doc_id = meta.get("document_id") or meta.get("source_document")
            page = meta.get("page", meta.get("page_start"))
            section = meta.get("section") or meta.get("section_heading")
            span = meta.get("character_span")
            if span is None and meta.get("char_start") is not None:
                span = [meta.get("char_start"), meta.get("char_end")]
            bbox = meta.get("bounding_box") or (
                (meta.get("bounding_boxes") or [None])[0]
            )
            grounding["fields"] = {
                "document_id": doc_id,
                "page": page,
                "page_start": meta.get("page_start"),
                "page_end": meta.get("page_end"),
                "section": section,
                "character_span": span,
                "bounding_box": bbox,
            }
            complete = True
            for key, val in [
                ("document_id", doc_id),
                ("page", page),
                ("section", section),
                ("character_span", span),
                ("bounding_box", bbox),
            ]:
                if val in (None, "", [], {}):
                    field_missing[key] += 1
                    grounding["issues"].append(f"missing_{key}")
                    if key != "section":  # section optional-ish but we still track
                        if key in {"document_id", "page", "character_span", "bounding_box"}:
                            complete = False
            if complete and section:
                n_complete += 1
            if q.get("page_overlap"):
                n_page_ok += 1
            elif expected and page is not None:
                grounding["issues"].append(
                    f"page_mismatch_got_{page}_expected_{sorted(expected)}"
                )
        elif hit:
            grounding["issues"].append("hit_without_matched_chunk_metadata")
        else:
            grounding["issues"].append("retrieval_miss")

        # Completeness across all Top-K (reviewer traceability)
        top_complete = 0
        for r in q.get("top_results") or []:
            m = r.get("metadata") or {}
            has_doc = bool(m.get("document_id") or m.get("source_document"))
            has_page = m.get("page") is not None or m.get("page_start") is not None
            has_span = m.get("character_span") is not None or m.get("char_start") is not None
            has_bbox = bool(m.get("bounding_box") or m.get("bounding_boxes"))
            if has_doc and has_page and has_span and has_bbox:
                top_complete += 1
        grounding["top_k_fully_grounded"] = top_complete
        grounding["top_k"] = len(q.get("top_results") or [])
        rows.append(grounding)

    n = max(len(per_query), 1)
    return {
        "summary": {
            "n_queries": len(per_query),
            "n_retrieval_hits": n_hits,
            "page_agreement_among_hits": (n_page_ok / n_hits) if n_hits else 0.0,
            "fully_grounded_matched_hits": (n_complete / n_hits) if n_hits else 0.0,
            "field_missing_on_matched_hits": field_missing,
            "mean_top_k_fully_grounded": sum(r["top_k_fully_grounded"] for r in rows)
            / n,
        },
        "per_query": rows,
    }
