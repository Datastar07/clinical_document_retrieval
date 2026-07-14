#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import add_device_arg, apply_device
from clinical_retrieval.retrieval.factory import build_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks for a query")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--query", required=True)
    parser.add_argument("--query-id", default="adhoc")
    parser.add_argument("--top-k", type=int, default=None)
    add_device_arg(parser)
    args = parser.parse_args()

    config = AppConfig.from_yaml(args.config).resolve(Path.cwd())
    apply_device(config, args.device)
    if args.top_k:
        config.retrieval.final_top_k = args.top_k
    retriever = build_retriever(config)
    result = retriever.retrieve(args.query, query_id=args.query_id)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
