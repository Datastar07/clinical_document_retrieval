# Reviewer runbook — how to run the full system

Step-by-step guide for running this project on **your machine**.  
Architecture: [`README.md`](README.md) · Outputs explained: [`OUTPUTS.md`](OUTPUTS.md) · Design: [`DESIGN.md`](DESIGN.md).

---

## 0) Prerequisites

| Need | Notes |
|------|--------|
| **Python 3.10+** | Required |
| **Git** | Clone the repo |
| **Docker** | For Qdrant (dense vectors). Host port **6340** |
| **RAM** | ≥16 GB recommended (embedding + reranker) |
| **Disk** | Several GB for HF models + processed ~1k-page chart |
| **GPU** | Optional; `DEVICE=auto` uses CUDA if present else CPU |
| **Tesseract** | Optional; better OCR on scanned pages |

Ship includes the sample clinical PDF under `data/raw/` and the evaluation JSON under `data/evaluation/`.

---

## 1) One-time install

```bash
git clone https://github.com/Datastar07/clinical_document_retrieval.git
cd clinical_document_retrieval

export PYTHONPATH=src
export TRANSFORMERS_NO_TF=1
export USE_TF=0

python3 -m pip install -e ".[dev,api]"
make qdrant
curl -s http://127.0.0.1:6340/readyz   # should succeed when Qdrant is up
```

Optional visual stack (only if you will build ColQwen):

```bash
python3 -m pip install -e ".[visual]"
```

Re-export `PYTHONPATH` / `TRANSFORMERS_NO_TF` / `USE_TF` in every new terminal (or add them to your shell profile).

---

## Case A — Assignment evaluation (reproduce Recall@10)

Full offline path for Tasks 1–3 grading.

```bash
make ingest
make index-novisual DEVICE=auto
make evaluate DEVICE=auto
```

| Step | What it does |
|------|----------------|
| `ingest` | Validate PDF → extract/OCR/layout → structure → grounded chunks |
| `index-novisual` | Build BM25 + dense (Qdrant) + structured store |
| `evaluate` | For each eval query: retrieve Top-10 (query text only) → metrics |

| Output | Contents |
|--------|----------|
| `outputs/evaluation_summary.json` | Recall@10, Hit@K, MRR, nDCG, missed queries, latency |
| `outputs/retrieval_results.json` | Per-query Top-10 (chunk, score, metadata) — large / gitignored |
| `outputs/grounding_report.json` | Page / section / span / bbox completeness |

**Expected:** Recall@10 = **1.0** (18/18).

CPU-only:

```bash
make ingest
make index-novisual DEVICE=cpu
make evaluate DEVICE=cpu
```

---

## Case B — Reviewer UI on the bundled chart

Requires Case A indexes (or equivalent ingest + index).

```bash
make api DEVICE=auto
```

| URL | Purpose |
|-----|---------|
| **http://127.0.0.1:9006/** | UI — search + optional grounded answer |
| http://127.0.0.1:9006/docs | Swagger |
| http://127.0.0.1:9006/health | Ready flag, device, active chart, pipeline job |

UI steps:

1. Confirm **indexes ready** and device pill (cuda / cpu).  
2. Mode: **Retrieve only** or **Retrieve + grounded answer**.  
3. Top-K (e.g. 10).  
4. Enter a query or click an example → **Search**.  
5. Inspect hits: page · section · span · bbox · score.

Curl smoke test:

```bash
curl -s http://127.0.0.1:9006/health | python -m json.tool

curl -s http://127.0.0.1:9006/retrieve \
  -H 'Content-Type: application/json' \
  -d '{"query":"What is the patient'"'"'s blood type?","top_k":5}'
```

Stop with `Ctrl+C` in the API terminal.

---

## Case C — Upload any clinical PDF (automatic full pipeline)

Runs **ingest → index → ready** in the background. No manual Make targets for that PDF.

1. `make qdrant` (once) and `make api DEVICE=auto`.  
2. Open http://127.0.0.1:9006/  
3. **Upload chart PDF** → choose `.pdf`.  
4. Optional Document ID.  
5. Leave **Build visual index** unchecked for speed (BM25 + dense + structured).  
6. **Upload & process** → wait until status = **ready**.  
7. Search as in Case B.

Upload **replaces** the previous active chart.

Via API:

```bash
curl -s http://127.0.0.1:9006/upload \
  -F 'file=@/absolute/path/to/chart.pdf' \
  -F 'document_id=MY-CHART-001' \
  -F 'build_visual=false'

curl -s http://127.0.0.1:9006/pipeline/<job_id> | python -m json.tool
# wait until "status": "ready"
```

Notes:

- Progress may sit near **15%** for a long time during ingest on ~1000-page charts — that is expected.  
- Retrieve/answer return busy while the job is running — wait for **ready**.  
- Check **Build visual index** only if you want ColQwen (much slower; GPU recommended).

---

## Case D — GPU vs CPU

| Goal | Example |
|------|---------|
| Auto (recommended) | `make api DEVICE=auto` · `make evaluate DEVICE=auto` |
| Force CUDA | `make api DEVICE=cuda` |
| Force CPU | `make api DEVICE=cpu` |

Without Make:

```bash
python scripts/serve_api.py --device auto --port 9006
python scripts/serve_api.py --device cpu --host 0.0.0.0 --port 9006
CLINICAL_DEVICE=cpu make index-novisual
```

First start downloads/loads models (can take several minutes). Later queries reuse warm models.

---

## Case E — Visual index on vs off

| Situation | How |
|-----------|-----|
| Default / gold eval / fast UI upload | **Off** — `make index-novisual` or upload without visual checkbox |
| Scans, forms, layout demos | **On** — `make index DEVICE=cuda` or upload with **Build visual index** |
| Serve without loading visual | `python scripts/serve_api.py --no-visual --device auto` |

Gold Recall@10 = 1.0 does **not** require visual indexing.

---

## Case F — Single query from CLI (no browser)

Indexes must already exist (Case A or finished Case C against config paths).

```bash
make retrieve DEVICE=auto QUERY='What is the patient'\''s blood type?'
make answer   DEVICE=auto QUERY='What is the patient'\''s blood type?'
```

`answer` defaults to **extractive** (no API key). Optional LLMs:

```bash
# OPENAI_API_KEY=... python scripts/answer.py --query '...' --provider openai --device auto --no-visual
# python scripts/answer.py --query '...' --provider ollama --model llama3.1 --device cpu --no-visual
```

---

## Case G — Ablations, grounding audit, latency

```bash
make ablate DEVICE=auto      # → outputs/ablation_summary.json
make evaluate DEVICE=auto    # if not already run
make grounding               # → outputs/grounding_report.json (needs eval details)
make latency DEVICE=auto     # → outputs/latency_profile.json (full pipeline only)
```

How to read these reports: [`DESIGN.md`](DESIGN.md) Scenario 1.

---

## Case H — Point the CLI at another PDF (no UI)

1. Copy PDF into `data/raw/`.  
2. Set `document.source_pdf`, `document_id`, `patient_id`, `patient_name` in `configs/default.yaml`.  
3. Rebuild and serve:

```bash
make ingest
make index-novisual DEVICE=auto
make api DEVICE=auto
```

Or ingest path override:

```bash
python scripts/ingest.py --config configs/default.yaml --input /path/to/chart.pdf
make index-novisual DEVICE=auto
```

---

## Quick decision table

| I want to… | Run |
|------------|-----|
| Reproduce Recall@10 | **Case A** |
| Use the web UI on the sample chart | **A → B** |
| Try my own PDF end-to-end | **Case C** |
| No GPU | **Case D** with `DEVICE=cpu` |
| Faster index | **Case E** without visual |
| One question in terminal | **Case F** |
| Prove hybrid > single channel | **Case G** (`make ablate`) |
| Wire a PDF via config only | **Case H** |

---

## Troubleshooting

| Symptom | What to do |
|---------|------------|
| `ready_for_retrieve: false` | Finish Case A ingest+index, or wait for Case C `ready` |
| Qdrant / dense errors | `make qdrant`; confirm **http://127.0.0.1:6340** |
| Upload stuck around 15% | Large PDF ingest; poll `/pipeline/{job_id}`; wait |
| CUDA out of memory | `DEVICE=cpu` and/or skip visual |
| Weak OCR on scans | Install Tesseract; keep OCR enabled in config |
| Port 9006 busy | `python scripts/serve_api.py --port 9010 --device auto` |
| Models “hanging” on first run | First Hugging Face download; needs network |
