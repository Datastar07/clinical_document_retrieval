"""Runtime device selection: auto | cuda | cpu.

Two approaches for users:
  1) Explicit: pass --device cuda|cpu (or CLINICAL_DEVICE / models.embedding_device)
  2) Auto: use CUDA when available, otherwise CPU fallback
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

DeviceChoice = Literal["auto", "cuda", "cpu"]


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(preference: str | None = None, *, log: bool = True) -> str:
    """
    Resolve effective torch device string ("cuda" or "cpu").

    Preference order:
      1) explicit function arg
      2) CLINICAL_DEVICE env
      3) "auto"
    """
    raw = (preference or os.environ.get("CLINICAL_DEVICE") or "auto").strip().lower()
    if raw in {"gpu"}:
        raw = "cuda"
    if raw not in {"auto", "cuda", "cpu"}:
        raise ValueError(f"Invalid device '{preference}'. Use auto|cuda|cpu.")

    has_cuda = cuda_available()
    if raw == "cpu":
        device = "cpu"
        reason = "explicit cpu"
    elif raw == "cuda":
        if has_cuda:
            device = "cuda"
            reason = "explicit cuda"
        else:
            device = "cpu"
            reason = "explicit cuda requested but unavailable → cpu fallback"
    else:  # auto
        device = "cuda" if has_cuda else "cpu"
        reason = "auto (cuda available)" if has_cuda else "auto (no cuda → cpu)"

    if log:
        logger.info("Device resolved: %s (%s)", device, reason)
        print(f"[device] using {device} ({reason})")
    return device


def apply_device(config, preference: str | None = None, *, log: bool = True):
    """Mutate AppConfig.models.embedding_device to the resolved device; return config."""
    device = resolve_device(preference if preference is not None else config.models.embedding_device, log=log)
    config.models.embedding_device = device
    return config


def add_device_arg(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device",
        default=None,
        choices=["auto", "cuda", "cpu", "gpu"],
        help=(
            "Compute device for embeddings/rerank/visual: "
            "auto (default; cuda if available else cpu), cuda, or cpu. "
            "Overrides configs models.embedding_device / CLINICAL_DEVICE."
        ),
    )
    return parser


def preference_from_args(args) -> str | None:
    """Return CLI --device if set, else None (config/env/auto apply)."""
    return getattr(args, "device", None)
