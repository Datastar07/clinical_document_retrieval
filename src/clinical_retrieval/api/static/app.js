/* Clinical retrieval reviewer UI — talks to same-origin FastAPI */

const $ = (id) => document.getElementById(id);

const els = {
  form: $("searchForm"),
  uploadForm: $("uploadForm"),
  pdfFile: $("pdfFile"),
  docId: $("docId"),
  buildVisual: $("buildVisual"),
  uploadBtn: $("uploadBtn"),
  pipelineBox: $("pipelineBox"),
  query: $("query"),
  mode: $("mode"),
  topK: $("topK"),
  runBtn: $("runBtn"),
  examples: $("examples"),
  results: $("results"),
  empty: $("empty"),
  error: $("error"),
  loading: $("loading"),
  answerBox: $("answerBox"),
  metaBox: $("metaBox"),
  healthPill: $("healthPill"),
  docPill: $("docPill"),
  devicePill: $("devicePill"),
};

let pollTimer = null;
let pipelineBusy = false;

function show(el, on = true) {
  el.classList.toggle("hidden", !on);
}

function setBusy(busy) {
  els.runBtn.disabled = busy || pipelineBusy;
  show(els.loading, busy);
  if (busy) {
    show(els.error, false);
    show(els.empty, false);
  }
}

function setUploadBusy(busy) {
  els.uploadBtn.disabled = busy;
  els.runBtn.disabled = busy || false;
  pipelineBusy = busy;
}

function truncate(text, n = 520) {
  const t = (text || "").trim();
  if (t.length <= n) return t;
  return t.slice(0, n - 1) + "…";
}

function fmtBBox(bb) {
  if (!Array.isArray(bb) || bb.length < 4) return null;
  return bb.map((x) => Number(x).toFixed(1)).join(", ");
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderPipeline(job) {
  if (!job) {
    show(els.pipelineBox, false);
    return;
  }
  const pct = Math.max(0, Math.min(100, Number(job.progress_pct) || 0));
  const status = job.status || "queued";
  els.pipelineBox.className = `pipeline ${status === "ready" ? "ready" : ""} ${
    status === "failed" ? "failed" : ""
  }`;
  const detail = job.error || job.detail || status;
  const doc = job.document_id || "—";
  els.pipelineBox.innerHTML = `
    <strong>${escapeHtml(status)}</strong>
    <div class="bar" aria-hidden="true"><span style="width:${pct}%"></span></div>
    <p class="meta">
      <span>${pct}%</span>
      <span>doc ${escapeHtml(doc)}</span>
      <span>${escapeHtml(String(detail))}</span>
      ${job.job_id ? `<span>job ${escapeHtml(job.job_id)}</span>` : ""}
    </p>
  `;
  show(els.pipelineBox, true);
}

async function refreshHealth() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    const ready = !!data.ready_for_retrieve;
    const job = data.pipeline;
    const busy = job && ["queued", "saving", "ingesting", "indexing"].includes(job.status);

    if (busy) {
      els.healthPill.textContent = `building · ${job.status}`;
      els.healthPill.className = "pill";
      pipelineBusy = true;
      els.runBtn.disabled = true;
      renderPipeline(job);
    } else {
      els.healthPill.textContent = ready ? "indexes ready" : "indexes missing";
      els.healthPill.className = `pill ${ready ? "ok" : "bad"}`;
      if (!pollTimer) {
        pipelineBusy = false;
        els.runBtn.disabled = false;
      }
      if (job) renderPipeline(job);
    }

    const active = data.active_document || {};
    els.docPill.textContent = `chart ${active.document_id || active.patient_name || "—"}`;

    const d = data.device || {};
    const resolved = d.resolved || "—";
    const cuda = d.cuda_available ? "cuda ok" : "cpu only";
    els.devicePill.textContent = `device ${resolved} · ${cuda}`;
  } catch (err) {
    els.healthPill.textContent = "api offline";
    els.healthPill.className = "pill bad";
    els.devicePill.textContent = "device —";
    els.docPill.textContent = "chart —";
  }
}

async function pollJob(jobId) {
  if (pollTimer) clearInterval(pollTimer);
  setUploadBusy(true);

  const tick = async () => {
    try {
      const res = await fetch(`/pipeline/${jobId}`);
      const job = await res.json();
      if (!res.ok) throw new Error(job.detail || "Poll failed");
      renderPipeline(job);

      if (job.status === "ready") {
        clearInterval(pollTimer);
        pollTimer = null;
        setUploadBusy(false);
        await refreshHealth();
        els.empty.textContent =
          "Chart indexed. Run a query to see ranked evidence with grounding metadata.";
        show(els.empty, true);
        return;
      }
      if (job.status === "failed") {
        clearInterval(pollTimer);
        pollTimer = null;
        setUploadBusy(false);
        els.error.textContent = job.error || "Pipeline failed";
        show(els.error, true);
        await refreshHealth();
        return;
      }
    } catch (err) {
      clearInterval(pollTimer);
      pollTimer = null;
      setUploadBusy(false);
      els.error.textContent = String(err.message || err);
      show(els.error, true);
    }
  };

  await tick();
  pollTimer = setInterval(tick, 2000);
}

async function runUpload(event) {
  event.preventDefault();
  const file = els.pdfFile.files && els.pdfFile.files[0];
  if (!file) return;

  show(els.error, false);
  setUploadBusy(true);
  renderPipeline({
    status: "queued",
    progress_pct: 3,
    detail: "Uploading PDF…",
    document_id: els.docId.value || file.name,
  });

  const body = new FormData();
  body.append("file", file);
  if (els.docId.value.trim()) body.append("document_id", els.docId.value.trim());
  body.append("build_visual", els.buildVisual.checked ? "true" : "false");

  try {
    const res = await fetch("/upload", { method: "POST", body });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText || "Upload failed");
    renderPipeline(data.job || { job_id: data.job_id, status: data.status });
    await pollJob(data.job_id);
  } catch (err) {
    setUploadBusy(false);
    els.error.textContent = String(err.message || err);
    show(els.error, true);
    renderPipeline({
      status: "failed",
      progress_pct: 100,
      error: String(err.message || err),
    });
  }
}

async function loadExamples() {
  try {
    const res = await fetch("/examples");
    const data = await res.json();
    const queries = data.queries || [];
    els.examples.innerHTML = "";
    queries.forEach((q) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "chip";
      btn.textContent = q;
      btn.addEventListener("click", () => {
        els.query.value = q;
        els.query.focus();
      });
      els.examples.appendChild(btn);
    });
  } catch (_) {
    /* optional */
  }
}

function renderAnswer(generation) {
  if (!generation) {
    show(els.answerBox, false);
    return;
  }
  const cites = (generation.citations_used || []).join(", ") || "—";
  els.answerBox.innerHTML = `
    <h2>Grounded answer</h2>
    <pre>${escapeHtml(generation.answer || "")}</pre>
    <p class="chunk-id" style="margin-top:0.7rem">
      provider=${escapeHtml(generation.provider || "")}
      · model=${escapeHtml(generation.model || "")}
      · citations=${escapeHtml(cites)}
      · grounded=${generation.grounded ? "yes" : "no"}
    </p>
  `;
  show(els.answerBox, true);
}

function renderMeta(payload, startedAt) {
  const ms = Math.round(performance.now() - startedAt);
  const n = (payload.retrieval?.results || payload.results || []).length;
  els.metaBox.innerHTML = `
    <h2>Run summary</h2>
    <dl>
      <div><dt>Results</dt><dd>${n}</dd></div>
      <div><dt>Latency</dt><dd>${ms} ms</dd></div>
      <div><dt>Mode</dt><dd>${escapeHtml(els.mode.value)}</dd></div>
      <div><dt>Pipeline</dt><dd>full</dd></div>
    </dl>
  `;
  show(els.metaBox, true);
}

function renderHits(results) {
  els.results.innerHTML = "";
  if (!results || !results.length) {
    show(els.empty, true);
    return;
  }
  show(els.empty, false);
  results.forEach((r, idx) => {
    const m = r.metadata || {};
    const page = m.page ?? m.page_start ?? "—";
    const section = m.section || m.section_heading || "—";
    const span = Array.isArray(m.character_span)
      ? `[${m.character_span.join(", ")}]`
      : "—";
    const bbox = fmtBBox(m.bounding_box);
    const card = document.createElement("article");
    card.className = "hit";
    card.style.animationDelay = `${idx * 40}ms`;
    card.innerHTML = `
      <div class="hit-head">
        <div class="rank">#${r.rank ?? idx + 1}</div>
        <div class="score">${Number(r.score ?? 0).toFixed(4)}</div>
      </div>
      <div class="ground">
        <span class="tag">page ${escapeHtml(String(page))}</span>
        <span class="tag">${escapeHtml(String(section))}</span>
        <span class="tag">span ${escapeHtml(String(span))}</span>
        ${bbox ? `<span class="tag">bbox ${escapeHtml(bbox)}</span>` : ""}
      </div>
      <p class="hit-body">${escapeHtml(truncate(r.content || ""))}</p>
      <div class="chunk-id">${escapeHtml(r.chunk_id || "")}</div>
    `;
    els.results.appendChild(card);
  });
}

async function runSearch(event) {
  event.preventDefault();
  const query = els.query.value.trim();
  if (!query) return;

  setBusy(true);
  show(els.answerBox, false);
  show(els.metaBox, false);
  els.results.innerHTML = "";

  const body = {
    query,
    top_k: Number(els.topK.value) || 10,
    profile: "full",
    no_visual: false,
    query_id: "ui",
  };

  const started = performance.now();
  try {
    const endpoint = els.mode.value === "answer" ? "/answer" : "/retrieve";
    if (endpoint === "/answer") {
      body.provider = "extractive";
    }
    const res = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || res.statusText || "Request failed");
    }

    const retrieval = data.retrieval || data;
    const results = retrieval.results || [];
    renderHits(results);
    renderMeta(data, started);
    if (data.generation) renderAnswer(data.generation);
  } catch (err) {
    els.error.textContent = String(err.message || err);
    show(els.error, true);
    show(els.empty, true);
  } finally {
    setBusy(false);
  }
}

els.form.addEventListener("submit", runSearch);
els.uploadForm.addEventListener("submit", runUpload);
refreshHealth();
loadExamples();
setInterval(refreshHealth, 5000);
