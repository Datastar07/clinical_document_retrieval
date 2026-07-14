"""FastAPI surface for clinical retrieval + grounded answers.

Reviewer quick start:
  pip install -e ".[api]"
  make index-novisual   # once, if indexes are missing
  make api              # → http://127.0.0.1:9006/docs
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import apply_device, cuda_available, resolve_device
from clinical_retrieval.generation.answer import bundle_to_dict, generate_answer
from clinical_retrieval.retrieval.factory import build_retriever

APP_VERSION = "2.0.0"
DEFAULT_CONFIG = os.environ.get("CLINICAL_CONFIG", "configs/default.yaml")
# Reviewer-friendly defaults: no visual model unless explicitly enabled
DEFAULT_PROFILE = os.environ.get("CLINICAL_API_PROFILE", "api")
DEFAULT_NO_VISUAL = os.environ.get("CLINICAL_API_NO_VISUAL", "1").strip().lower() in {
    "1",
    "true",
    "yes",
}
DEFAULT_DEVICE = os.environ.get("CLINICAL_DEVICE", "auto")

app = FastAPI(
    title="Clinical Document Retrieval API",
    version=APP_VERSION,
    description=(
        "Hybrid multimodal retrieval over clinical PDFs with evidence grounding "
        "(page / section / span / bbox). Use `/docs` for interactive try-out."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(..., examples=["What is the patient's blood type?"])
    query_id: str = "api"
    top_k: int = Field(10, ge=1, le=50)
    profile: Literal["api", "full"] | None = Field(
        default=None,
        description="api = faster (no visual); full = max recall. Default from env/server.",
    )
    no_visual: bool | None = Field(
        default=None,
        description="Force-disable ColQwen visual channel. Default true for easy reviewer runs.",
    )


class AnswerRequest(QueryRequest):
    provider: Literal[
        "extractive",
        "openai",
        "anthropic",
        "ollama",
        "openai_compatible",
        "vllm",
        "local",
    ] = Field(
        default="extractive",
        description="extractive needs no API key; openai/anthropic/ollama optional.",
    )
    model: str | None = None
    max_tokens: int = Field(512, ge=64, le=4096)


def _project_root() -> Path:
    # Prefer cwd (repo root when started via make api); else package parents
    cwd = Path.cwd()
    if (cwd / "configs" / "default.yaml").exists():
        return cwd
    return Path(__file__).resolve().parents[3]


def _load_config(profile: str, no_visual: bool, device_pref: str | None = None) -> AppConfig:
    root = _project_root()
    cfg_path = Path(DEFAULT_CONFIG)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    config = AppConfig.from_yaml(cfg_path).resolve(root)
    config.retrieval.profile = profile
    if no_visual or profile == "api":
        config.retrieval.enable_visual = False
        config.models.visual_enabled = False
    apply_device(config, device_pref or DEFAULT_DEVICE, log=True)
    return config


def _index_status(config: AppConfig) -> dict[str, Any]:
    index_dir = Path(config.paths.index_dir)
    processed = Path(config.paths.processed_dir)
    return {
        "chunks_jsonl": (processed / "chunks.jsonl").exists(),
        "bm25": (index_dir / "bm25.pkl").exists(),
        "dense_cache": (index_dir / "dense_embeddings.npy").exists(),
        "sqlite": Path(config.paths.sqlite_path).exists(),
        "processed_dir": str(processed),
        "index_dir": str(index_dir),
    }


@lru_cache(maxsize=8)
def get_retriever(profile: str = "api", no_visual: bool = True, device_pref: str = "auto"):
    config = _load_config(profile, no_visual, device_pref)
    status = _index_status(config)
    if not status["chunks_jsonl"] or not status["bm25"]:
        raise RuntimeError(
            "Indexes missing. From repo root run: "
            "`make ingest` (if needed) then `make index-novisual` "
            f"(chunks={status['chunks_jsonl']}, bm25={status['bm25']})."
        )
    return build_retriever(
        config,
        load_visual=not no_visual and profile != "api",
        device=config.models.embedding_device,
    )


def _resolve_flags(req: QueryRequest) -> tuple[str, bool, str]:
    profile = req.profile or DEFAULT_PROFILE
    no_visual = DEFAULT_NO_VISUAL if req.no_visual is None else bool(req.no_visual)
    if profile == "api":
        no_visual = True
    device_pref = DEFAULT_DEVICE
    return profile, no_visual, device_pref


@app.get("/", tags=["meta"])
def root():
    resolved = resolve_device(DEFAULT_DEVICE, log=False)
    return {
        "service": "clinical-document-retrieval",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /retrieve": "Top-K grounded chunks",
            "POST /answer": "Retrieve + grounded extractive/LLM answer",
            "GET /examples": "Sample queries",
        },
        "defaults": {
            "profile": DEFAULT_PROFILE,
            "no_visual": DEFAULT_NO_VISUAL,
            "device_preference": DEFAULT_DEVICE,
            "device_resolved": resolved,
            "cuda_available": cuda_available(),
            "config": DEFAULT_CONFIG,
        },
    }


@app.get("/health", tags=["meta"])
def health():
    try:
        config = _load_config(DEFAULT_PROFILE, DEFAULT_NO_VISUAL, DEFAULT_DEVICE)
        status = _index_status(config)
        ready = bool(status["chunks_jsonl"] and status["bm25"])
        return {
            "status": "ok" if ready else "degraded",
            "version": APP_VERSION,
            "ready_for_retrieve": ready,
            "device": {
                "preference": DEFAULT_DEVICE,
                "resolved": config.models.embedding_device,
                "cuda_available": cuda_available(),
            },
            "indexes": status,
            "hint": None
            if ready
            else "Run `make index-novisual DEVICE=cpu` (and ingest if chunks missing) before /retrieve.",
        }
    except Exception as exc:
        return {"status": "error", "version": APP_VERSION, "detail": str(exc)}


@app.get("/examples", tags=["meta"])
def examples():
    return {
        "queries": [
            "What is the patient's blood type?",
            "What allergies does the patient have?",
            "Who electronically signed the encounter on February 20, 2024?",
            "What was the HbA1c when an SGLT2 inhibitor was added?",
        ],
        "curl_retrieve": (
            "curl -s http://127.0.0.1:9006/retrieve "
            "-H 'Content-Type: application/json' "
            '-d \'{"query":"What is the patient\'\\"\'\\"\'s blood type?","top_k":5}\''
        ),
        "device_flags": {
            "cli": "python scripts/serve_api.py --device auto|cuda|cpu",
            "env": "CLINICAL_DEVICE=auto|cuda|cpu",
        },
    }


@app.post("/retrieve", tags=["retrieval"])
def retrieve(req: QueryRequest):
    profile, no_visual, device_pref = _resolve_flags(req)
    try:
        retriever = get_retriever(profile, no_visual, device_pref)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load retriever: {exc}") from exc

    retriever.config.final_top_k = req.top_k
    return retriever.retrieve(req.query, query_id=req.query_id).model_dump()


@app.post("/answer", tags=["generation"])
def answer(req: AnswerRequest):
    profile, no_visual, device_pref = _resolve_flags(req)
    try:
        retriever = get_retriever(profile, no_visual, device_pref)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    retriever.config.final_top_k = req.top_k
    result = retriever.retrieve(req.query, query_id=req.query_id)
    try:
        bundle = generate_answer(
            req.query,
            result,
            provider=req.provider,
            model=req.model,
            max_evidence=req.top_k,
            max_tokens=req.max_tokens,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return {
        "retrieval": result.model_dump(),
        "generation": bundle_to_dict(bundle),
        "device": resolve_device(device_pref, log=False),
    }
