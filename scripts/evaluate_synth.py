#!/usr/bin/env python3
"""Mini-eval across synthetic document types → generalization_report.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import add_device_arg, apply_device
from clinical_retrieval.evaluation.evaluator import load_evaluation, run_evaluation
from clinical_retrieval.indexing.bm25_index import BM25Index
from clinical_retrieval.indexing.structured_store import StructuredStore
from clinical_retrieval.pipeline import load_chunks
from clinical_retrieval.retrieval.hybrid_retriever import HybridRetriever
from clinical_retrieval.retrieval.reranker import Reranker


class _NullDense:
    def search(self, query: str, top_k: int = 50):
        return []


def _build_local_retriever(processed: Path, indexes: Path, config: AppConfig, use_rerank: bool):
    chunks = load_chunks(processed / "chunks.jsonl")
    bm25 = BM25Index.load(indexes / "bm25.pkl")
    sqlite = indexes / "clinical_meta.db"
    structured = StructuredStore(sqlite) if sqlite.exists() else None
    posting = structured.load_posting_index() if structured else None
    # Lexical + structured + exact only (no shared Qdrant / visual for synth isolation)
    cfg = config.retrieval.model_copy(deep=True)
    cfg.enable_dense = False
    cfg.enable_visual = False
    cfg.profile = "api"
    reranker = None
    if use_rerank and config.models.use_reranker:
        try:
            reranker = Reranker(
                model_name=config.models.reranker_model,
                device=config.models.embedding_device,
            )
        except Exception as exc:
            print(f"Reranker unavailable for synth eval: {exc}")
    return HybridRetriever(
        chunks,
        bm25,
        _NullDense(),
        cfg,
        reranker=reranker,
        structured=structured,
        visual=None,
        posting_index=posting,
        lexicon_path=config.structure.lexicon_path,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", default="data/synth/manifest.json")
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--output", default="outputs/generalization_report.json")
    add_device_arg(ap)
    args = ap.parse_args()

    root = Path.cwd()
    base = AppConfig.from_yaml(args.config).resolve(root)
    apply_device(base, args.device)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    per_doc = []
    hit_bar = 0.8

    for doc in manifest["documents"]:
        doc_id = doc["id"]
        processed = root / "data" / "synth" / doc_id / "processed"
        indexes = root / "data" / "synth" / doc_id / "indexes"
        eval_path = root / doc["evaluation_path"]
        if not (processed / "chunks.jsonl").exists():
            per_doc.append({"id": doc_id, "status": "not_ingested", "hit_at_10": 0.0})
            continue
        items = load_evaluation(eval_path)
        retriever = _build_local_retriever(
            processed, indexes, base, use_rerank=not args.no_rerank
        )
        report = run_evaluation(
            retriever,
            items,
            token_threshold=base.evaluation.token_overlap_threshold,
            fuzzy_threshold=base.evaluation.fuzzy_threshold,
        )
        summary = report["summary"]
        per_doc.append(
            {
                "id": doc_id,
                "status": "ok",
                "n_queries": len(items),
                "hit_at_10": summary.get("hit_at_10", 0.0),
                "hit_at_1": summary.get("hit_at_1", 0.0),
                "mrr_at_10": summary.get("mrr_at_10", 0.0),
                "missed_queries": summary.get("missed_queries", []),
                "pass_bar": summary.get("hit_at_10", 0.0) >= hit_bar,
                "expected_parser": doc.get("expected_parser"),
            }
        )
        print(
            f"{doc_id}: Hit@10={summary.get('hit_at_10', 0):.3f} "
            f"MRR={summary.get('mrr_at_10', 0):.3f}"
        )

    overall = {
        "hit_bar": hit_bar,
        "n_docs": len(per_doc),
        "docs_passing": sum(1 for d in per_doc if d.get("pass_bar")),
        "mean_hit_at_10": (
            sum(d.get("hit_at_10", 0) for d in per_doc) / max(len(per_doc), 1)
        ),
        "documents": per_doc,
    }
    out = root / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
