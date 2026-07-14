"""FastAPI surface for clinical retrieval + grounded answers + PDF upload pipeline.

Reviewer quick start:
  pip install -e ".[api]"
  make api              # → http://127.0.0.1:9006/  (UI)  ·  /docs (Swagger)
  Upload a PDF in the UI — ingest + index runs automatically.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from clinical_retrieval.api.pipeline_jobs import pipeline_jobs
from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import apply_device, cuda_available, resolve_device
from clinical_retrieval.generation.answer import bundle_to_dict, generate_answer
from clinical_retrieval.retrieval.factory import build_retriever
from clinical_retrieval.retrieval.model_registry import clear_registry

APP_VERSION = "2.1.0"
DEFAULT_CONFIG = os.environ.get("CLINICAL_CONFIG", "configs/default.yaml")
DEFAULT_PROFILE = "full"
DEFAULT_NO_VISUAL = os.environ.get("CLINICAL_API_NO_VISUAL", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
DEFAULT_DEVICE = os.environ.get("CLINICAL_DEVICE", "auto")
# Uploads skip ColQwen by default (slow). Set CLINICAL_UPLOAD_VISUAL=1 to build visual.
DEFAULT_UPLOAD_SKIP_VISUAL = os.environ.get("CLINICAL_UPLOAD_VISUAL", "0").strip().lower() not in {
    "1",
    "true",
    "yes",
}

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_UPLOAD_MB = int(os.environ.get("CLINICAL_UPLOAD_MAX_MB", "250"))

app = FastAPI(
    title="Clinical Document Retrieval API",
    version=APP_VERSION,
    description=(
        "Hybrid multimodal retrieval over clinical PDFs with evidence grounding "
        "(page / section / span / bbox). Open `/` for the reviewer UI or `/docs` for Swagger. "
        "POST `/upload` runs ingest + index automatically."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class QueryRequest(BaseModel):
    query: str = Field(..., examples=["What is the patient's blood type?"])
    query_id: str = "api"
    top_k: int = Field(10, ge=1, le=50)
    profile: Literal["api", "full"] | None = Field(
        default="full",
        description="Ignored: API always uses the full pipeline.",
    )
    no_visual: bool | None = Field(
        default=None,
        description="Force-disable ColQwen visual channel. Default false (visual on).",
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
    cwd = Path.cwd()
    if (cwd / "configs" / "default.yaml").exists():
        return cwd
    return Path(__file__).resolve().parents[3]


def _apply_active_document(config: AppConfig) -> AppConfig:
    overlay = pipeline_jobs.active_document()
    if not overlay:
        return config
    if overlay.get("source_pdf"):
        config.document.source_pdf = overlay["source_pdf"]
    if overlay.get("document_id"):
        config.document.document_id = overlay["document_id"]
        config.document.patient_id = overlay.get("patient_id") or overlay["document_id"]
    if overlay.get("patient_name"):
        config.document.patient_name = overlay["patient_name"]
    return config


def _load_config(profile: str, no_visual: bool, device_pref: str | None = None) -> AppConfig:
    root = _project_root()
    cfg_path = Path(DEFAULT_CONFIG)
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    config = AppConfig.from_yaml(cfg_path).resolve(root)
    config = _apply_active_document(config)
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


def _invalidate_retriever() -> None:
    clear_registry()
    get_retriever.cache_clear()


pipeline_jobs.set_on_ready(_invalidate_retriever)


@lru_cache(maxsize=8)
def get_retriever(profile: str = "full", no_visual: bool = False, device_pref: str = "auto"):
    config = _load_config(profile, no_visual, device_pref)
    status = _index_status(config)
    if not status["chunks_jsonl"] or not status["bm25"]:
        raise RuntimeError(
            "Indexes missing. Upload a PDF in the UI, or run `make ingest` then "
            f"`make index-novisual` (chunks={status['chunks_jsonl']}, bm25={status['bm25']})."
        )
    return build_retriever(
        config,
        load_visual=not no_visual and profile != "api",
        device=config.models.embedding_device,
    )


def _resolve_flags(req: QueryRequest) -> tuple[str, bool, str]:
    profile = "full"
    no_visual = DEFAULT_NO_VISUAL if req.no_visual is None else bool(req.no_visual)
    device_pref = DEFAULT_DEVICE
    return profile, no_visual, device_pref


@app.get("/", tags=["ui"], include_in_schema=False)
def ui_home():
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="UI static files missing")
    return FileResponse(index)


@app.get("/info", tags=["meta"])
def info():
    resolved = resolve_device(DEFAULT_DEVICE, log=False)
    return {
        "service": "clinical-document-retrieval",
        "version": APP_VERSION,
        "ui": "/",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "POST /retrieve": "Top-K grounded chunks",
            "POST /answer": "Retrieve + grounded extractive/LLM answer",
            "POST /upload": "Upload PDF → auto ingest + index",
            "GET /pipeline/{job_id}": "Pipeline job status",
            "GET /pipeline": "Latest pipeline job",
            "GET /examples": "Sample queries",
        },
        "active_document": pipeline_jobs.active_document(),
        "defaults": {
            "profile": DEFAULT_PROFILE,
            "no_visual": DEFAULT_NO_VISUAL,
            "upload_skip_visual": DEFAULT_UPLOAD_SKIP_VISUAL,
            "device_preference": DEFAULT_DEVICE,
            "device_resolved": resolved,
            "cuda_available": cuda_available(),
            "config": DEFAULT_CONFIG,
            "max_upload_mb": MAX_UPLOAD_MB,
        },
    }


@app.get("/health", tags=["meta"])
def health():
    try:
        config = _load_config(DEFAULT_PROFILE, DEFAULT_NO_VISUAL, DEFAULT_DEVICE)
        status = _index_status(config)
        ready = bool(status["chunks_jsonl"] and status["bm25"])
        latest = pipeline_jobs.latest()
        return {
            "status": "ok" if ready else "degraded",
            "version": APP_VERSION,
            "ready_for_retrieve": ready,
            "active_document": pipeline_jobs.active_document()
            or {
                "document_id": config.document.document_id,
                "source_pdf": config.document.source_pdf,
                "patient_name": config.document.patient_name,
            },
            "pipeline": latest.to_dict() if latest else None,
            "device": {
                "preference": DEFAULT_DEVICE,
                "resolved": config.models.embedding_device,
                "cuda_available": cuda_available(),
            },
            "indexes": status,
            "hint": None
            if ready
            else "Upload a PDF in the UI, or run `make ingest` + `make index-novisual`.",
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


@app.post("/upload", tags=["pipeline"])
async def upload_pdf(
    file: UploadFile = File(..., description="Clinical PDF to ingest + index"),
    document_id: str | None = Form(default=None),
    patient_name: str | None = Form(default=None),
    build_visual: str = Form(
        default="false",
        description="true/false — also build ColQwen visual index (slow / GPU).",
    ),
):
    """Save PDF and run full ingest → index automatically in the background."""
    filename = file.filename or "upload.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty upload")
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(data)} bytes). Max {MAX_UPLOAD_MB} MB.",
        )
    if not data.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="File does not look like a PDF")

    want_visual = str(build_visual).strip().lower() in {"1", "true", "yes", "on"}
    skip_visual = not want_visual
    if DEFAULT_NO_VISUAL:
        skip_visual = True

    root = _project_root()
    raw_dir = root / "data" / "raw"

    def load_cfg() -> AppConfig:
        return _load_config(DEFAULT_PROFILE, skip_visual, DEFAULT_DEVICE)

    try:
        job = pipeline_jobs.start(
            pdf_bytes=data,
            filename=filename,
            document_id=document_id,
            patient_name=patient_name,
            skip_visual=skip_visual,
            load_config=load_cfg,
            raw_dir=raw_dir,
            device_pref=DEFAULT_DEVICE,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "job_id": job.job_id,
        "status": job.status,
        "message": "Upload accepted. Poll GET /pipeline/{job_id} until status=ready.",
        "poll_url": f"/pipeline/{job.job_id}",
        "job": job.to_dict(),
    }


@app.get("/pipeline", tags=["pipeline"])
def pipeline_latest():
    job = pipeline_jobs.latest()
    if not job:
        return {"job": None, "busy": False}
    return {"job": job.to_dict(), "busy": pipeline_jobs.is_busy()}


@app.get("/pipeline/{job_id}", tags=["pipeline"])
def pipeline_status(job_id: str):
    job = pipeline_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Unknown job_id={job_id}")
    return job.to_dict()


@app.post("/retrieve", tags=["retrieval"])
def retrieve(req: QueryRequest):
    if pipeline_jobs.is_busy():
        raise HTTPException(
            status_code=409,
            detail="Pipeline is building indexes from an upload. Wait until status=ready.",
        )
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
    if pipeline_jobs.is_busy():
        raise HTTPException(
            status_code=409,
            detail="Pipeline is building indexes from an upload. Wait until status=ready.",
        )
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
