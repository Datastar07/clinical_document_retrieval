#!/usr/bin/env python3
"""Profile full-pipeline retrieve latency on the gold evaluation set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import add_device_arg, apply_device
from clinical_retrieval.evaluation.evaluator import load_evaluation
from clinical_retrieval.retrieval.factory import build_retriever
from clinical_retrieval.retrieval.model_registry import clear_registry


def _timed_run(config: AppConfig, items, load_visual: bool) -> dict:
    clear_registry()
    t0 = time.perf_counter()
    retriever = build_retriever(config, load_visual=load_visual)
    load_s = time.perf_counter() - t0
    latencies = []
    for item in items:
        q = item.query if hasattr(item, "query") else item["query"]
        qid = item.query_id if hasattr(item, "query_id") else item.get("query_id", "Q")
        t1 = time.perf_counter()
        retriever.retrieve(q, query_id=str(qid))
        latencies.append(time.perf_counter() - t1)
    latencies.sort()
    n = len(latencies) or 1
    return {
        "n_queries": len(latencies),
        "device": config.models.embedding_device,
        "model_load_sec": round(load_s, 3),
        "mean_sec": round(sum(latencies) / n, 3),
        "p50_sec": round(latencies[n // 2], 3),
        "p95_sec": round(latencies[min(n - 1, int(n * 0.95))], 3),
        "max_sec": round(max(latencies), 3) if latencies else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--output", default="outputs/latency_profile.json")
    add_device_arg(ap)
    args = ap.parse_args()

    root = Path.cwd()
    full = AppConfig.from_yaml(args.config).resolve(root)
    apply_device(full, args.device)
    full.retrieval.profile = "full"
    items = load_evaluation(full.paths.evaluation_path)[: args.limit]

    print("Profiling full pipeline …")
    stats = _timed_run(full, items, load_visual=bool(full.models.visual_enabled))

    out = {
        "limit": args.limit,
        "profile": "full",
        **stats,
    }
    path = root / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
