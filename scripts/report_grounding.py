#!/usr/bin/env python3
"""Rebuild grounding_report.json from existing evaluation_details.json (no re-retrieve)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_retrieval.evaluation.grounding_report import build_grounding_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--details", default="outputs/evaluation_details.json")
    ap.add_argument("--output", default="outputs/grounding_report.json")
    args = ap.parse_args()

    details = json.loads(Path(args.details).read_text(encoding="utf-8"))
    report = build_grounding_report(details)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
