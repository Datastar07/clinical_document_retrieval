#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import add_device_arg, apply_device
from clinical_retrieval.evaluation.evaluator import (
    load_evaluation,
    run_evaluation,
    write_error_analysis,
)
from clinical_retrieval.evaluation.grounding_report import build_grounding_report
from clinical_retrieval.retrieval.factory import build_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against JSON dataset")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--evaluation", default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-visual", action="store_true")
    add_device_arg(parser)
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config).resolve(Path.cwd())
    apply_device(config, args.device)
    if args.evaluation:
        config.paths.evaluation_path = str(Path(args.evaluation).resolve())
    if args.output:
        config.paths.outputs_dir = str(Path(args.output).resolve())
    if args.top_k:
        config.retrieval.final_top_k = args.top_k
        config.evaluation.top_k = args.top_k
    if args.no_visual:
        config.retrieval.enable_visual = False
        config.models.visual_enabled = False

    out_dir = Path(config.paths.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_evaluation(config.paths.evaluation_path)
    print(f"Loaded {len(items)} evaluation queries")
    retriever = build_retriever(config, load_visual=not args.no_visual)
    report = run_evaluation(
        retriever,
        items,
        token_threshold=config.evaluation.token_overlap_threshold,
        fuzzy_threshold=config.evaluation.fuzzy_threshold,
    )

    with open(out_dir / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(report["summary"], f, indent=2)
    with open(out_dir / "retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "query_id": q["query_id"],
                    "query": q["query"],
                    "results": q["top_results"],
                }
                for q in report["per_query"]
            ],
            f,
            indent=2,
            ensure_ascii=False,
        )
    with open(out_dir / "evaluation_details.json", "w", encoding="utf-8") as f:
        json.dump(report["per_query"], f, indent=2, ensure_ascii=False)
    write_error_analysis(report["per_query"], out_dir / "error_analysis.csv")
    grounding = build_grounding_report(report["per_query"])
    with open(out_dir / "grounding_report.json", "w", encoding="utf-8") as f:
        json.dump(grounding, f, indent=2, ensure_ascii=False)

    s = report["summary"]
    print("\n=== Evaluation Summary ===")
    print(f"Recall@10 / Hit@10: {s['hit_at_10']:.3f}")
    print(f"Hit@1: {s['hit_at_1']:.3f} | Hit@3: {s['hit_at_3']:.3f} | Hit@5: {s['hit_at_5']:.3f}")
    print(f"MRR@10: {s['mrr_at_10']:.3f} | nDCG@10: {s['ndcg_at_10']:.3f}")
    print(f"Precision@10: {s.get('precision_at_10', 0):.3f}")
    print(f"Page Hit@10: {s.get('page_hit_at_10', 0):.3f}")
    print(f"Page overlap accuracy: {s['page_overlap_accuracy']:.3f}")
    print(f"Duplicate ratio: {s.get('duplicate_result_ratio', 0):.3f}")
    print(f"Missed: {s['missed_queries']}")
    gs = grounding["summary"]
    print("\n=== Grounding Summary ===")
    print(f"Page agreement among hits: {gs['page_agreement_among_hits']:.3f}")
    print(f"Fully grounded matched hits: {gs['fully_grounded_matched_hits']:.3f}")
    print(f"Mean Top-K fully grounded: {gs['mean_top_k_fully_grounded']:.2f}")
    print(f"Outputs written to {out_dir}")


if __name__ == "__main__":
    main()
