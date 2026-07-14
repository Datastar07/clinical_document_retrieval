#!/usr/bin/env python3
"""Worker loop: claim queued ingest jobs and run the document pipeline.

Production: replace FileJobQueue with Redis/SQS and run N workers behind a K8s
Deployment. This stub demonstrates concurrency-safe claim → process → complete/fail.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from clinical_retrieval.config import AppConfig
from clinical_retrieval.ingestion.job_queue import FileJobQueue
from clinical_retrieval.pipeline import run_ingest


def _config_for_job(base_cfg: Path, job, work_root: Path) -> AppConfig:
    raw = yaml.safe_load(base_cfg.read_text(encoding="utf-8")) or {}
    doc_dir = work_root / job.document_id / job.job_id
    processed = doc_dir / "processed"
    indexes = doc_dir / "indexes"
    processed.mkdir(parents=True, exist_ok=True)
    indexes.mkdir(parents=True, exist_ok=True)
    raw.setdefault("document", {})
    raw["document"]["source_pdf"] = job.source_pdf
    raw["document"]["document_id"] = job.document_id
    raw["document"]["patient_id"] = job.document_id
    raw["document"]["processing_version"] = job.processing_version
    raw.setdefault("paths", {})
    raw["paths"]["processed_dir"] = str(processed)
    raw["paths"]["index_dir"] = str(indexes)
    raw["paths"]["page_images_dir"] = str(processed / "page_images")
    raw["paths"]["sqlite_path"] = str(indexes / "clinical_meta.db")
    # Keep workers light by default
    raw.setdefault("extraction", {})
    raw["extraction"]["render_pages"] = raw["extraction"].get("render_pages", False)
    raw.setdefault("structure", {})
    raw["structure"]["parser"] = raw["structure"].get("parser", "auto")
    raw.setdefault("models", {})
    raw["models"]["visual_enabled"] = False
    raw.setdefault("chunking", {})
    raw["chunking"]["create_page_visual_chunks"] = False
    for k, v in (job.config_overlay or {}).items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k].update(v)
        else:
            raw[k] = v
    cfg_path = doc_dir / "job_config.yaml"
    cfg_path.write_text(yaml.dump(raw), encoding="utf-8")
    return AppConfig.from_yaml(cfg_path).resolve(Path.cwd())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--queue-dir", default="data/job_queue")
    ap.add_argument("--work-root", default="data/job_work")
    ap.add_argument("--once", action="store_true", help="Process at most one job then exit")
    ap.add_argument("--poll-sec", type=float, default=2.0)
    args = ap.parse_args()

    queue = FileJobQueue(args.queue_dir)
    work_root = Path(args.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    base_cfg = Path(args.config)

    print(f"Worker started. queue={args.queue_dir} stats={queue.stats()}")
    while True:
        job = queue.claim_next()
        if job is None:
            if args.once:
                print("No jobs.")
                return
            time.sleep(args.poll_sec)
            continue
        print(f"Claimed {job.job_id} document={job.document_id} attempt={job.attempts}")
        try:
            config = _config_for_job(base_cfg, job, work_root)
            summary = run_ingest(config)
            queue.complete(job, summary)
            print(json.dumps({"completed": job.job_id, "summary": {
                "pages": summary.get("pages"),
                "chunks": summary.get("chunks"),
                "parser": summary.get("structure_parser"),
            }}, indent=2))
        except Exception as exc:
            queue.fail(job, str(exc))
            print(f"Failed {job.job_id}: {exc}")
        if args.once:
            return


if __name__ == "__main__":
    main()
