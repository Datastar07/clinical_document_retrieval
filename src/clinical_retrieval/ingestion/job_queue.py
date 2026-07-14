from __future__ import annotations

"""File-backed ingest job queue (swap Redis/SQS in production)."""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class IngestJob:
    job_id: str
    document_id: str
    source_pdf: str
    status: str = "queued"  # queued | running | succeeded | failed | dead
    attempts: int = 0
    max_attempts: int = 3
    processing_version: str = "2.0.0"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str | None = None
    result_summary: dict[str, Any] | None = None
    config_overlay: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IngestJob":
        return cls(**data)


class FileJobQueue:
    """
    Simple durable queue under a directory:

      queue_dir/
        queued/{job_id}.json
        running/
        succeeded/
        failed/
        dead/
    """

    def __init__(self, queue_dir: str | Path):
        self.root = Path(queue_dir)
        for name in ("queued", "running", "succeeded", "failed", "dead"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    def enqueue(
        self,
        *,
        document_id: str,
        source_pdf: str,
        processing_version: str = "2.0.0",
        config_overlay: dict[str, Any] | None = None,
        max_attempts: int = 3,
    ) -> IngestJob:
        job = IngestJob(
            job_id=str(uuid.uuid4()),
            document_id=document_id,
            source_pdf=str(source_pdf),
            processing_version=processing_version,
            config_overlay=config_overlay or {},
            max_attempts=max_attempts,
        )
        self._write("queued", job)
        return job

    def claim_next(self) -> IngestJob | None:
        queued = sorted((self.root / "queued").glob("*.json"))
        if not queued:
            return None
        path = queued[0]
        data = json.loads(path.read_text(encoding="utf-8"))
        job = IngestJob.from_dict(data)
        job.status = "running"
        job.attempts += 1
        job.updated_at = time.time()
        path.unlink(missing_ok=True)
        self._write("running", job)
        return job

    def complete(self, job: IngestJob, summary: dict[str, Any] | None = None) -> None:
        self._remove("running", job.job_id)
        job.status = "succeeded"
        job.result_summary = summary
        job.updated_at = time.time()
        job.error = None
        self._write("succeeded", job)

    def fail(self, job: IngestJob, error: str) -> None:
        self._remove("running", job.job_id)
        job.error = error
        job.updated_at = time.time()
        if job.attempts >= job.max_attempts:
            job.status = "dead"
            self._write("dead", job)
        else:
            job.status = "queued"
            self._write("queued", job)

    def stats(self) -> dict[str, int]:
        return {
            name: len(list((self.root / name).glob("*.json")))
            for name in ("queued", "running", "succeeded", "failed", "dead")
        }

    def _write(self, bucket: str, job: IngestJob) -> None:
        path = self.root / bucket / f"{job.job_id}.json"
        path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

    def _remove(self, bucket: str, job_id: str) -> None:
        path = self.root / bucket / f"{job_id}.json"
        path.unlink(missing_ok=True)
