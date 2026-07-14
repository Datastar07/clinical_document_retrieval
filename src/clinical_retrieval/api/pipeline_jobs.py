"""Background upload → ingest → index jobs for the reviewer UI/API."""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from clinical_retrieval.config import AppConfig
from clinical_retrieval.device import apply_device
from clinical_retrieval.indexing.build import run_build_index
from clinical_retrieval.pipeline import run_ingest
from clinical_retrieval.retrieval.model_registry import clear_registry


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_document_id(value: str) -> str:
    stem = Path(value).stem if value else "upload"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-._")
    return (cleaned or "upload")[:80]


@dataclass
class PipelineJob:
    job_id: str
    status: str = "queued"  # queued|saving|ingesting|indexing|ready|failed
    stage: str = "queued"
    detail: str | None = None
    progress_pct: int = 0
    document_id: str | None = None
    source_pdf: str | None = None
    filename: str | None = None
    error: str | None = None
    skip_visual: bool = True
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    finished_at: str | None = None
    ingest_summary: dict[str, Any] | None = None
    index_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PipelineJobManager:
    """Single-flight background ingest+index (replaces active chart)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, PipelineJob] = {}
        self._current_id: str | None = None
        self._active_document: dict[str, str] | None = None
        self._on_ready: Callable[[], None] | None = None

    def set_on_ready(self, cb: Callable[[], None]) -> None:
        self._on_ready = cb

    def active_document(self) -> dict[str, str] | None:
        with self._lock:
            return dict(self._active_document) if self._active_document else None

    def set_active_document(self, overlay: dict[str, str]) -> None:
        with self._lock:
            self._active_document = dict(overlay)

    def get(self, job_id: str) -> PipelineJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job

    def latest(self) -> PipelineJob | None:
        with self._lock:
            if self._current_id and self._current_id in self._jobs:
                return self._jobs[self._current_id]
            if not self._jobs:
                return None
            return max(self._jobs.values(), key=lambda j: j.created_at)

    def is_busy(self) -> bool:
        with self._lock:
            if not self._current_id:
                return False
            job = self._jobs.get(self._current_id)
            return bool(job and job.status in {"queued", "saving", "ingesting", "indexing"})

    def _update(self, job: PipelineJob, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            setattr(job, k, v)
        job.updated_at = _utc_now()

    def start(
        self,
        *,
        pdf_bytes: bytes,
        filename: str,
        document_id: str | None,
        patient_name: str | None,
        skip_visual: bool,
        load_config: Callable[[], AppConfig],
        raw_dir: Path,
        device_pref: str,
    ) -> PipelineJob:
        if self.is_busy():
            raise RuntimeError("A pipeline job is already running. Wait for it to finish.")

        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF uploads are supported")
        doc_id = sanitize_document_id(document_id or filename)

        job = PipelineJob(
            job_id=uuid.uuid4().hex[:12],
            status="queued",
            stage="queued",
            document_id=doc_id,
            filename=filename,
            skip_visual=skip_visual,
            progress_pct=2,
            detail="Queued for ingest + index",
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._current_id = job.job_id

        thread = threading.Thread(
            target=self._run,
            kwargs={
                "job_id": job.job_id,
                "pdf_bytes": pdf_bytes,
                "filename": filename,
                "document_id": doc_id,
                "patient_name": patient_name or doc_id,
                "skip_visual": skip_visual,
                "load_config": load_config,
                "raw_dir": raw_dir,
                "device_pref": device_pref,
            },
            daemon=True,
            name=f"pipeline-{job.job_id}",
        )
        thread.start()
        return job

    def _run(
        self,
        *,
        job_id: str,
        pdf_bytes: bytes,
        filename: str,
        document_id: str,
        patient_name: str,
        skip_visual: bool,
        load_config: Callable[[], AppConfig],
        raw_dir: Path,
        device_pref: str,
    ) -> None:
        with self._lock:
            job = self._jobs[job_id]

        try:
            self._update(job, status="saving", stage="saving", progress_pct=5, detail="Saving PDF")
            raw_dir.mkdir(parents=True, exist_ok=True)
            dest = raw_dir / f"{document_id}.pdf"
            dest.write_bytes(pdf_bytes)
            self._update(job, source_pdf=str(dest))

            config = load_config()
            apply_device(config, device_pref, log=True)
            config.document.source_pdf = str(dest)
            config.document.document_id = document_id
            config.document.patient_id = document_id
            config.document.patient_name = patient_name
            if skip_visual:
                config.models.visual_enabled = False
                config.retrieval.enable_visual = False

            self.set_active_document(
                {
                    "source_pdf": str(dest),
                    "document_id": document_id,
                    "patient_id": document_id,
                    "patient_name": patient_name,
                }
            )

            self._update(
                job,
                status="ingesting",
                stage="ingesting",
                progress_pct=15,
                detail="Extracting pages, structure, and chunks",
            )
            ingest_summary = run_ingest(config)
            self._update(
                job,
                ingest_summary={
                    k: ingest_summary.get(k)
                    for k in (
                        "pages",
                        "encounters",
                        "sections",
                        "chunks",
                        "structure_parser",
                    )
                    if k in ingest_summary
                },
                progress_pct=55,
                detail=f"Ingested {ingest_summary.get('chunks', '?')} chunks",
            )

            self._update(
                job,
                status="indexing",
                stage="indexing",
                progress_pct=60,
                detail="Building BM25 + dense + structured indexes",
            )

            def on_index(stage: str, detail: str | None = None) -> None:
                pct_map = {
                    "loading_chunks": 62,
                    "bm25": 68,
                    "sqlite": 74,
                    "dense": 82,
                    "visual": 92,
                    "visual_skip": 90,
                    "done": 98,
                }
                self._update(
                    job,
                    stage=f"indexing:{stage}",
                    progress_pct=pct_map.get(stage, job.progress_pct),
                    detail=detail or stage,
                )

            index_summary = run_build_index(
                config,
                skip_visual=skip_visual,
                skip_dense=False,
                progress_cb=on_index,
            )

            clear_registry()
            if self._on_ready:
                self._on_ready()

            self._update(
                job,
                status="ready",
                stage="ready",
                progress_pct=100,
                detail="Ready to search",
                index_summary={
                    "n_chunks": index_summary.get("n_chunks"),
                    "visual": index_summary.get("visual"),
                    "structured": index_summary.get("structured"),
                    "skip_visual": index_summary.get("skip_visual"),
                },
                finished_at=_utc_now(),
            )
        except Exception as exc:
            self._update(
                job,
                status="failed",
                stage="failed",
                progress_pct=100,
                error=str(exc),
                detail=str(exc),
                finished_at=_utc_now(),
            )


# Process-wide singleton used by FastAPI
pipeline_jobs = PipelineJobManager()
