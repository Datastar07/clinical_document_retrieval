#!/usr/bin/env python3
"""Start the Clinical Retrieval FastAPI server for local review.

Usage:
  python scripts/serve_api.py
  python scripts/serve_api.py --device cuda
  python scripts/serve_api.py --device cpu --host 0.0.0.0 --port 9006
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("CLINICAL_API_HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("CLINICAL_API_PORT", "9006")))
    ap.add_argument("--reload", action="store_true")
    ap.add_argument("--config", default=os.environ.get("CLINICAL_CONFIG", "configs/default.yaml"))
    ap.add_argument(
        "--profile",
        default=os.environ.get("CLINICAL_API_PROFILE", "api"),
        choices=["api", "full"],
    )
    ap.add_argument(
        "--device",
        default=os.environ.get("CLINICAL_DEVICE", "auto"),
        choices=["auto", "cuda", "cpu", "gpu"],
        help="auto|cuda|cpu — auto uses CUDA if available else CPU",
    )
    ap.add_argument(
        "--visual",
        action="store_true",
        help="Enable visual channel (slow / needs GPU + visual index).",
    )
    args = ap.parse_args()

    root = Path.cwd()
    os.environ["CLINICAL_CONFIG"] = str(Path(args.config))
    os.environ["CLINICAL_API_PROFILE"] = args.profile
    os.environ["CLINICAL_API_NO_VISUAL"] = "0" if args.visual else "1"
    os.environ["CLINICAL_DEVICE"] = "cuda" if args.device == "gpu" else args.device
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("PYTHONPATH", "src")

    src = str((root / "src").resolve())
    if src not in sys.path:
        sys.path.insert(0, src)

    from clinical_retrieval.device import resolve_device

    resolved = resolve_device(os.environ["CLINICAL_DEVICE"])

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit(
            "uvicorn/fastapi missing. Install with: pip install -e '.[api]'"
        ) from exc

    print(f"Starting API on http://{args.host}:{args.port}")
    print(f"  Swagger UI: http://{args.host}:{args.port}/docs")
    print(f"  Health:     http://{args.host}:{args.port}/health")
    print(
        f"  profile={args.profile}  device={resolved} "
        f"(pref={args.device})  no_visual={not args.visual}  config={args.config}"
    )

    uvicorn.run(
        "clinical_retrieval.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        factory=False,
    )


if __name__ == "__main__":
    main()
