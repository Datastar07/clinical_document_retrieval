#!/usr/bin/env python3
"""Retrieve Top-K evidence and optionally generate a grounded LLM answer.

Providers:
  extractive          — no LLM (default, always works)
  openai              — closed-source (OPENAI_API_KEY)
  anthropic           — closed-source (ANTHROPIC_API_KEY)
  ollama / openai_compatible / vllm / local — open-source OpenAI-compatible endpoint
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import add_device_arg, apply_device
from clinical_retrieval.generation.answer import bundle_to_dict, generate_answer
from clinical_retrieval.retrieval.factory import build_retriever


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--query", required=True)
    ap.add_argument("--query-id", default="answer")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument(
        "--provider",
        default="extractive",
        choices=[
            "extractive",
            "openai",
            "anthropic",
            "ollama",
            "openai_compatible",
            "vllm",
            "local",
        ],
    )
    ap.add_argument("--model", default=None)
    ap.add_argument("--no-visual", action="store_true")
    ap.add_argument("--profile", default=None, help="api|full")
    ap.add_argument("--output", default=None, help="Write JSON answer bundle")
    add_device_arg(ap)
    args = ap.parse_args()

    config = AppConfig.from_yaml(args.config).resolve(Path.cwd())
    apply_device(config, args.device)
    config.retrieval.final_top_k = args.top_k
    if args.profile:
        config.retrieval.profile = args.profile
    if args.no_visual or (args.profile or "").lower() == "api":
        config.retrieval.enable_visual = False
        config.models.visual_enabled = False

    retriever = build_retriever(config, load_visual=not args.no_visual)
    result = retriever.retrieve(args.query, query_id=args.query_id)
    bundle = generate_answer(
        args.query,
        result,
        provider=args.provider,
        model=args.model,
        max_evidence=args.top_k,
    )
    payload = {
        "retrieval": result.model_dump(),
        "generation": bundle_to_dict(bundle),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
