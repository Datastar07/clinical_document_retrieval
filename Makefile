.PHONY: setup setup-api ingest index evaluate ablate test all qdrant latency rebuild-sqlite answer enqueue-ingest ingest-worker grounding api api-reload

PYTHON ?= python3
DEVICE ?= auto
export PYTHONPATH := src:$(PYTHONPATH)
export TRANSFORMERS_NO_TF := 1
export USE_TF := 0
export CLINICAL_DEVICE ?= $(DEVICE)
export HF_HOME ?= /root/clinical_artifacts/hf_cache

setup:
	$(PYTHON) -m pip install -e ".[dev]"

setup-api:
	$(PYTHON) -m pip install -e ".[dev,api]"

qdrant:
	docker compose up -d qdrant || true

# FastAPI — Reviewer UI http://HOST:9006/  ·  Swagger /docs
# Examples: make api DEVICE=cuda   |   make api DEVICE=cpu   |   make api DEVICE=auto
api:
	$(PYTHON) scripts/serve_api.py --host 0.0.0.0 --port 9006 --device $(DEVICE)

api-reload:
	$(PYTHON) scripts/serve_api.py --host 127.0.0.1 --port 9006 --reload --device $(DEVICE)

ingest:
	$(PYTHON) scripts/ingest.py --config configs/default.yaml

index:
	$(PYTHON) scripts/build_index.py --config configs/default.yaml --device $(DEVICE)

index-novisual:
	$(PYTHON) scripts/build_index.py --config configs/default.yaml --skip-visual --device $(DEVICE)

evaluate:
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml --top-k 10 --device $(DEVICE)

evaluate-novisual:
	$(PYTHON) scripts/evaluate.py --config configs/default.yaml --top-k 10 --no-visual --device $(DEVICE)

grounding:
	$(PYTHON) scripts/report_grounding.py --details outputs/evaluation_details.json

ablate:
	$(PYTHON) scripts/ablate.py --config configs/default.yaml --device $(DEVICE)

retrieve:
	$(PYTHON) scripts/retrieve.py --config configs/default.yaml --query "$(QUERY)" --device $(DEVICE)

answer:
	$(PYTHON) scripts/answer.py --config configs/default.yaml --query "$(QUERY)" --provider extractive --no-visual --profile api --device $(DEVICE)

enqueue-ingest:
	$(PYTHON) scripts/enqueue_ingest.py --pdf "$(PDF)" --document-id "$(DOC_ID)"

ingest-worker:
	$(PYTHON) scripts/ingest_worker.py --config configs/default.yaml --once

latency:
	$(PYTHON) scripts/profile_latency.py --config configs/default.yaml --limit 8 --device $(DEVICE)

rebuild-sqlite:
	$(PYTHON) -c "from pathlib import Path; from clinical_retrieval.config import AppConfig; from clinical_retrieval.indexing.structured_store import StructuredStore; from clinical_retrieval.pipeline import load_chunks, load_encounters; c=AppConfig.from_yaml('configs/default.yaml').resolve(Path.cwd()); st=StructuredStore(c.paths.sqlite_path); print(st.rebuild(load_chunks(Path(c.paths.processed_dir)/'chunks.jsonl'), load_encounters(Path(c.paths.processed_dir)/'encounters.json'))); st.close()"

test:
	$(PYTHON) -m pytest -q

all: ingest index evaluate
