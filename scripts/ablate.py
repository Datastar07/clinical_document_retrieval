#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import add_device_arg, apply_device
from clinical_retrieval.evaluation.evaluator import load_evaluation, run_evaluation
from clinical_retrieval.retrieval.factory import build_retriever


ABLATIONS = [
    ("bm25_only", dict(enable_bm25=True, enable_dense=False, enable_exact=False, enable_structured=False, enable_visual=False, use_reranker=False)),
    ("dense_only", dict(enable_bm25=False, enable_dense=True, enable_exact=False, enable_structured=False, enable_visual=False, use_reranker=False)),
    ("bm25_dense", dict(enable_bm25=True, enable_dense=True, enable_exact=False, enable_structured=False, enable_visual=False, use_reranker=False)),
    ("bm25_dense_structured", dict(enable_bm25=True, enable_dense=True, enable_exact=True, enable_structured=True, enable_visual=False, use_reranker=False)),
    ("bm25_dense_visual", dict(enable_bm25=True, enable_dense=True, enable_exact=True, enable_structured=False, enable_visual=True, use_reranker=False)),
    ("full_no_rerank", dict(enable_bm25=True, enable_dense=True, enable_exact=True, enable_structured=True, enable_visual=True, use_reranker=False)),
    ("full_with_rerank", dict(enable_bm25=True, enable_dense=True, enable_exact=True, enable_structured=True, enable_visual=True, use_reranker=True)),
]


def apply_flags(config: AppConfig, flags: dict) -> AppConfig:
    cfg = deepcopy(config)
    for k, v in flags.items():
        if k == "use_reranker":
            cfg.models.use_reranker = bool(v)
            if not v:
                cfg.retrieval.score_weights.reranker = 0.0
                cfg.retrieval.score_weights.hybrid = 0.75
                cfg.retrieval.score_weights.metadata = 0.25
        else:
            setattr(cfg.retrieval, k, bool(v))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval ablations")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-visual-load", action="store_true",
                        help="Do not load ColQwen (visual channel returns empty)")
    add_device_arg(parser)
    args = parser.parse_args()

    base = AppConfig.from_yaml(args.config).resolve(Path.cwd())
    apply_device(base, args.device)
    out_dir = Path(args.output) if args.output else Path(base.paths.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = load_evaluation(base.paths.evaluation_path)

    rows = []
    for name, flags in ABLATIONS:
        print(f"\n=== Ablation: {name} ===")
        cfg = apply_flags(base, flags)
        load_visual = bool(flags.get("enable_visual")) and not args.skip_visual_load
        retriever = build_retriever(cfg, load_visual=load_visual)
        report = run_evaluation(
            retriever,
            items,
            token_threshold=cfg.evaluation.token_overlap_threshold,
            fuzzy_threshold=cfg.evaluation.fuzzy_threshold,
        )
        summary = report["summary"]
        row = {"ablation": name, **{k: summary.get(k) for k in (
            "hit_at_1", "hit_at_5", "hit_at_10", "recall_at_10", "mrr_at_10",
            "ndcg_at_10", "precision_at_10", "page_hit_at_10", "avg_latency_sec",
        )}}
        rows.append(row)
        print(json.dumps(row, indent=2))

    with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"\nWrote {out_dir / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
