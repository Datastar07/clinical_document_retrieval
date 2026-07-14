#!/usr/bin/env python3
"""Ingest + light-index each synthetic document type; write per-doc summaries."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import yaml

from clinical_retrieval.config import AppConfig
from clinical_retrieval.indexing.bm25_index import BM25Index
from clinical_retrieval.indexing.structured_store import StructuredStore
from clinical_retrieval.pipeline import load_chunks, load_encounters, run_ingest


def _deep_update(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--manifest", default="data/synth/manifest.json")
    ap.add_argument("--skip-docling", action="store_true")
    args = ap.parse_args()

    root = Path.cwd()
    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    results = []

    for doc in manifest["documents"]:
        doc_id = doc["id"]
        pdf = root / doc["pdf"]
        if not pdf.exists():
            print(f"SKIP {doc_id}: missing {pdf}")
            results.append({"id": doc_id, "status": "missing_pdf"})
            continue

        out_processed = root / "data" / "synth" / doc_id / "processed"
        out_indexes = root / "data" / "synth" / doc_id / "indexes"
        if out_processed.exists():
            shutil.rmtree(out_processed)
        if out_indexes.exists():
            shutil.rmtree(out_indexes)
        out_processed.mkdir(parents=True, exist_ok=True)
        out_indexes.mkdir(parents=True, exist_ok=True)

        overlay = {
            "document": {
                "source_pdf": str(pdf),
                "document_id": f"SYNTH-{doc_id}",
                "patient_id": f"SYNTH-{doc_id}",
                "patient_name": doc_id,
            },
            "paths": {
                "processed_dir": str(out_processed),
                "index_dir": str(out_indexes),
                "page_images_dir": str(out_processed / "page_images"),
                "sqlite_path": str(out_indexes / "clinical_meta.db"),
                "evaluation_path": str(root / doc["evaluation_path"]),
            },
            "structure": {
                "parser": "auto",
                "docling_enabled": not args.skip_docling,
                "docling_mode": "inline",
            },
            "extraction": {
                "render_pages": False,
                "ocr_enabled": doc_id == "scanned_like",
            },
            "chunking": {"create_page_visual_chunks": False},
            "models": {"visual_enabled": False},
        }
        merged = _deep_update(base_cfg, overlay)
        cfg_path = out_processed / "run_config.yaml"
        cfg_path.write_text(yaml.dump(merged), encoding="utf-8")
        config = AppConfig.from_yaml(cfg_path).resolve(root)

        print(f"\n=== Ingest {doc_id} ===")
        summary = run_ingest(config)
        chunks = load_chunks(out_processed / "chunks.jsonl")
        encounters = load_encounters(out_processed / "encounters.json")
        BM25Index(chunks).save(out_indexes / "bm25.pkl")
        store = StructuredStore(config.paths.sqlite_path)
        stats = store.rebuild(chunks, encounters)
        store.close()

        selected = summary.get("structure_parser")
        expected = doc.get("expected_parser")
        ok_parser = True
        if expected == "synthetic_soap":
            ok_parser = selected == "synthetic_soap"
        elif expected == "generic":
            ok_parser = selected in {"generic", "docling"}
            # Must not force soap when ENC markers absent
            if selected == "synthetic_soap":
                ok_parser = False

        entry = {
            "id": doc_id,
            "status": "ok",
            "pages": summary.get("pages"),
            "chunks": summary.get("chunks"),
            "encounters": summary.get("encounters"),
            "structure_parser": selected,
            "expected_parser": expected,
            "parser_ok": ok_parser,
            "docling": summary.get("docling"),
            "sqlite": stats,
        }
        results.append(entry)
        print(json.dumps(entry, indent=2))

    out = root / "outputs" / "synth_ingest_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"documents": results}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
