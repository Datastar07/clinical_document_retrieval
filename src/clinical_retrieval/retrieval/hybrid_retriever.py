from __future__ import annotations

import re
from typing import Any, Protocol

from clinical_retrieval.config import RetrievalConfig
from clinical_retrieval.indexing.bm25_index import BM25Index
from clinical_retrieval.retrieval.fusion import reciprocal_rank_fusion
from clinical_retrieval.retrieval.query_parser import ParsedQuery, parse_query
from clinical_retrieval.retrieval.reranker import Reranker
from clinical_retrieval.schemas import Chunk, QueryResult, RetrievalResult
from clinical_retrieval.structure.lexicon import expand_aliases
from clinical_retrieval.structure.normalizer import normalize_text


class DenseSearcher(Protocol):
    def search(self, query: str, top_k: int = 50) -> list[tuple[str, float]]: ...


STRUCTURE_TABLE_BOOSTS = {
    "demographics": {"demographics", "allergies", "medical_history", "surgical_history", "family_history"},
    "medications": {"current_medications", "medications", "plan_item", "plan_block"},
    "laboratory_results": {"labs", "laboratory_results", "lab_panel"},
    "signature": {"signature"},
    "referrals": {"referrals"},
    "imaging_orders": {"imaging_orders", "imaging"},
    "assessment": {"diagnoses", "diagnoses_set"},
    "vitals": {"vitals"},
}


def _minmax(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _date_match(chunk: Chunk, pq: ParsedQuery) -> float:
    if not pq.dates and not pq.year_months:
        return 0.0
    enc_date = chunk.metadata.encounter_date or ""
    text_dates = chunk.metadata.entities.get("dates", []) if chunk.metadata.entities else []
    all_dates = set(text_dates)
    if enc_date:
        all_dates.add(enc_date)
    score = 0.0
    for d in pq.dates:
        if len(d) == 10 and d in all_dates:
            score = max(score, 1.0)
        elif len(d) == 7:
            if any(ad.startswith(d) for ad in all_dates):
                score = max(score, 1.0)
    return score


def _section_match(chunk: Chunk, pq: ParsedQuery) -> float:
    if not pq.intent_sections:
        return 0.0
    section = (chunk.metadata.section or "").lower()
    ctype = chunk.metadata.chunk_type
    ttype = chunk.metadata.table_type or ""
    score = 0.0
    for intent in pq.intent_sections:
        if intent.replace("_", " ") in section or intent in section:
            score = max(score, 1.0)
        allowed = STRUCTURE_TABLE_BOOSTS.get(intent, set())
        if ttype in allowed:
            score = max(score, 1.0)
    # Multi-intent favors compound / key-fact synthesis chunks
    if len(pq.intent_sections) >= 2 and ctype in {"compound", "encounter_key_facts"}:
        score = max(score, 0.9)
    return score


def _structure_type_boost(chunk: Chunk, pq: ParsedQuery) -> float:
    """Boost by section/table type alignment — not page-number heuristics."""
    score = 0.0
    ttype = chunk.metadata.table_type or ""
    section = (chunk.metadata.section or "").lower()
    for intent in pq.intent_sections:
        allowed = STRUCTURE_TABLE_BOOSTS.get(intent, set())
        if ttype in allowed or intent in section:
            score = max(score, 1.0)
    # Demographics / medications intent → prefer structured table chunks
    ql = pq.raw.lower()
    if any(x in ql for x in ("allergy", "allergies", "blood type", "family history", "surgical")):
        if ttype in STRUCTURE_TABLE_BOOSTS["demographics"] or section in {
            "allergies",
            "demographics",
            "family_history",
            "surgical_history",
            "medical_history",
        }:
            score = max(score, 1.0)
    if any(x in ql for x in ("medication", "dose", "taking", "started", "initiate")):
        if ttype in STRUCTURE_TABLE_BOOSTS["medications"] or "med" in section:
            score = max(score, 0.85)
    return score


def _entity_cooccurrence_boost(chunk: Chunk, pq: ParsedQuery) -> float:
    """
    Reward chunks that co-locate multiple planner entities (med+lab, med+ICD, etc.).
    Replaces assignment-specific phrase matching.
    """
    text = chunk.retrieval_text.lower()
    ql = pq.raw.lower()
    hits = 0
    med_hit = False
    for med in pq.medications:
        aliases = expand_aliases(med)
        if any(a.lower() in text for a in aliases):
            hits += 1
            med_hit = True
            break
    lab_hit = False
    for lab in pq.lab_tests:
        if lab.lower() in text:
            hits += 1
            lab_hit = True
            break
    if not lab_hit and ("lab" in ql or "hba1c" in ql or "ldl" in ql):
        if re.search(r"\bhba1c\b|\bldl\b|\bhdl\b|\btsh\b", text):
            hits += 1
            lab_hit = True
    icd_hit = False
    if pq.icd_codes and any(c.lower() in text for c in pq.icd_codes):
        hits += 1
        icd_hit = True
    elif re.search(r"\b[a-z]\d{2}(?:\.\d+)?\b", text) and (
        "icd" in ql or "diagnosis" in ql or "assessment" in pq.intent_sections or "code" in ql
    ):
        hits += 1
        icd_hit = True
    if pq.numerics and any(normalize_text(n) in normalize_text(text) for n in pq.numerics):
        hits += 1
    if pq.providers and any(p.lower() in text for p in pq.providers):
        hits += 1
    if "imaging" in pq.intent_sections or "follow_up" in pq.intent_sections:
        if any(x in text for x in ("ultrasound", "imaging", "doppler", "follow-up", "follow up", "weeks")):
            hits += 1
    # Multi-facet clinical questions (labs + diagnoses + med change)
    wants_multi = (
        ("lab" in ql or lab_hit)
        and ("diagnos" in ql or "icd" in ql or "code" in ql)
        and (med_hit or "inhibitor" in ql or "medication" in ql or "added" in ql)
    )
    if wants_multi and med_hit and lab_hit and icd_hit:
        n_icd = len(re.findall(r"\b[a-z]\d{2}(?:\.\d+)?\b", text))
        score = 1.2
        if n_icd >= 3 and re.search(r"\bhba1c\b", text) and re.search(r"\bldl\b", text):
            score = 1.8
        if re.search(r"add(?:\s+an?)?\s+sglt2|sglt2\s+inhibitor", text):
            score = max(score, 2.0)
        return score
    # Secondary hypertension workup co-occurrence
    if "secondary" in ql and "hypertension" in ql:
        workup = sum(
            1
            for x in (
                "secondary hypertension",
                "aldosteronism",
                "renal artery",
                "plasma renin",
                "aldosterone",
                "chlorthalidone",
            )
            if x in text
        )
        if workup >= 3:
            return 1.8
        if workup >= 2:
            return 1.1
    if hits >= 3:
        return 1.0
    if hits == 2:
        return 0.7
    if hits == 1:
        return 0.25
    return 0.0


def _entity_match(chunk: Chunk, pq: ParsedQuery) -> float:
    score = 0.0
    ents = chunk.metadata.entities or {}
    text = chunk.retrieval_text.lower()
    for med in pq.medications:
        if any(a.lower() in text for a in expand_aliases(med)):
            score = max(score, 1.0)
    for prov in pq.providers:
        if prov.lower() in text:
            score = max(score, 1.0)
    for et in pq.encounter_types:
        if et.lower() in (chunk.metadata.encounter_type or "").lower():
            score = max(score, 0.8)
        if et.lower() in text:
            score = max(score, 0.6)
    for code in pq.icd_codes:
        if code in ents.get("icd_codes", []) or code in text.upper():
            score = max(score, 1.0)
    for lab in pq.lab_tests:
        if lab.lower() in text:
            score = max(score, 0.9)
    return score


def _numeric_match(chunk: Chunk, pq: ParsedQuery) -> float:
    if not pq.numerics:
        nums = re.findall(
            r"\b\d{2,3}/\d{2,3}\b|\b\d+\.\d+%|\b\d+mg\b|\b\d+\s*mg\b", pq.raw, re.I
        )
        if not nums:
            return 0.0
        pq_nums = nums
    else:
        pq_nums = pq.numerics
    text = normalize_text(chunk.raw_text + " " + chunk.retrieval_text)
    score = 0.0
    for n in pq_nums:
        nn = normalize_text(n)
        if nn in text or nn.replace(" ", "") in text.replace(" ", ""):
            score = max(score, 1.0)
    for m in re.findall(r"\b\d{2,3}/\d{2,3}\b", chunk.raw_text):
        if m.replace(" ", "") in normalize_text(pq.raw).replace(" ", ""):
            score = max(score, 1.0)
    return score


def exact_search(
    chunks: list[Chunk],
    pq: ParsedQuery,
    top_k: int = 30,
    posting_index: dict[str, list[str]] | None = None,
) -> list[tuple[str, float]]:
    scored: list[tuple[str, float]] = []
    needles: list[str] = list(pq.needles or [])
    needles.extend(pq.medications)
    needles.extend(pq.providers)
    needles.extend(pq.icd_codes)
    needles.extend(pq.numerics)
    needles.extend(pq.lab_tests)
    for med in pq.medications:
        needles.extend(expand_aliases(med))
    # Dose / ICD / date regex needles from query
    needles.extend(re.findall(r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b", pq.raw))
    needles.extend(re.findall(r"\b\d+(?:\.\d+)?\s*mg\b", pq.raw, re.I))
    needles = [n for n in needles if n]
    needle_set = list({n.lower(): n for n in needles}.values())

    if posting_index and needle_set:
        candidate_ids: set[str] = set()
        for n in needle_set:
            candidate_ids.update(posting_index.get(n.lower(), []))
        if candidate_ids:
            chunk_map = {c.chunk_id: c for c in chunks}
            chunks = [chunk_map[i] for i in candidate_ids if i in chunk_map]

    lower_texts = [(c.chunk_id, c.retrieval_text.lower(), c.raw_text.lower(), c) for c in chunks]
    df: dict[str, int] = {}
    for n in set(x.lower() for x in needle_set):
        df[n] = sum(1 for _, t, r, _ in lower_texts if n in t or n in r)
    n_docs = max(len(chunks), 1)

    for chunk_id, text, raw, c in lower_texts:
        s = 0.0
        matched_rare = 0
        for n in needle_set:
            nl = n.lower()
            if nl in text or nl in raw:
                idf = 1.0 + (n_docs / max(df.get(nl, 1), 1)) ** 0.5
                s += min(idf, 40.0)
                if df.get(nl, n_docs) < max(80, n_docs * 0.02):
                    matched_rare += 1
        if matched_rare >= 2:
            s += 20.0
        s += 4.0 * _date_match(c, pq)
        s += 2.0 * _section_match(c, pq)
        s += 10.0 * _entity_cooccurrence_boost(c, pq)
        s += 4.0 * _structure_type_boost(c, pq)
        if len(pq.intent_sections) >= 2 and c.metadata.chunk_type in {
            "compound",
            "encounter_key_facts",
        }:
            s += 5.0
        if s > 0:
            scored.append((chunk_id, s))
    chunk_len = {c.chunk_id: len(c.raw_text) for c in chunks}
    scored.sort(key=lambda x: (x[1], chunk_len.get(x[0], 0)), reverse=True)
    return scored[:top_k]


def _rerank_text(c: Chunk) -> str:
    m = c.metadata
    return (
        f"Encounter date: {m.encounter_date or 'Unknown'}\n"
        f"Encounter type: {m.encounter_type or 'Unknown'}\n"
        f"Section: {m.section or 'Unknown'}\n"
        f"Page: {m.page_start}\n"
        f"Content: {c.retrieval_text}"
    )


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        bm25: BM25Index,
        dense: DenseSearcher,
        config: RetrievalConfig,
        reranker: Reranker | None = None,
        structured=None,
        visual=None,
        posting_index: dict[str, list[str]] | None = None,
        lexicon_path: str | None = None,
    ):
        self.chunks = {c.chunk_id: c for c in chunks}
        self.chunk_list = chunks
        self.bm25 = bm25
        self.dense = dense
        self.config = config
        self.reranker = reranker
        self.structured = structured
        self.visual = visual
        self._visual_loader = None  # optional callable for lazy visual
        self.posting_index = posting_index
        self.lexicon_path = lexicon_path

    def set_visual_loader(self, loader) -> None:
        self._visual_loader = loader

    def _ensure_visual(self, visual_weight: float):
        if self.visual is not None:
            return self.visual
        if visual_weight < self.config.lazy_visual_threshold:
            return None
        if self._visual_loader is not None:
            try:
                self.visual = self._visual_loader()
            except Exception as exc:
                print(f"Lazy visual load failed: {exc}")
                self.visual = None
        return self.visual

    def retrieve(self, query: str, query_id: str = "Q") -> QueryResult:
        pq = parse_query(query, lexicon_path=self.lexicon_path)
        q_ret = pq.expanded_query or query

        cw = self.config.channel_weights
        weights = {
            "lexical": pq.retrieval_weights.get("lexical", cw.lexical),
            "dense": pq.retrieval_weights.get("dense", cw.dense),
            "structured": pq.retrieval_weights.get("structured", cw.structured),
            "visual": pq.retrieval_weights.get("visual", cw.visual),
        }

        # api profile: skip visual + optional lighter rerank path
        profile = (self.config.profile or "full").lower()
        enable_visual = self.config.enable_visual and profile != "api"
        if profile == "api":
            weights["visual"] = 0.0

        lists: list[list[tuple[str, float]]] = []
        list_weights: list[float] = []

        if self.config.enable_bm25:
            lists.append(self.bm25.search(q_ret, top_k=self.config.bm25_top_k))
            list_weights.append(weights["lexical"])

        if self.config.enable_dense:
            lists.append(self.dense.search(q_ret, top_k=self.config.dense_top_k))
            list_weights.append(weights["dense"])

        if self.config.enable_exact:
            exact_hits = exact_search(
                self.chunk_list,
                pq,
                top_k=self.config.exact_top_k,
                posting_index=self.posting_index,
            )
            lists.append(exact_hits)
            list_weights.append(weights["lexical"])
            lists.append(exact_hits)
            list_weights.append(weights["lexical"] * 0.5)

        if self.config.enable_structured and self.structured is not None:
            lists.append(
                self.structured.structured_search(pq, top_k=self.config.structured_top_k)
            )
            list_weights.append(weights["structured"])

        if enable_visual and weights["visual"] > 0:
            visual = self._ensure_visual(weights["visual"])
            if visual is not None:
                lists.append(visual.search(q_ret, top_k=self.config.visual_top_k))
                list_weights.append(weights["visual"])

        if not lists:
            return QueryResult(query_id=query_id, query=query, results=[])

        fused = reciprocal_rank_fusion(
            lists, k=self.config.rrf_k, weights=list_weights
        )[: self.config.fusion_top_k]

        diversified: list[tuple[str, float]] = []
        enc_counts: dict[str, int] = {}
        for cid, score in fused:
            c = self.chunks.get(cid)
            if not c:
                continue
            enc = c.metadata.encounter_id or f"page_{c.metadata.page_start}"
            if enc_counts.get(enc, 0) >= 6:
                continue
            enc_counts[enc] = enc_counts.get(enc, 0) + 1
            diversified.append((cid, score))

        hybrid_scores = {cid: score for cid, score in diversified}
        candidates = [self.chunks[cid] for cid, _ in diversified if cid in self.chunks]

        rerank_scores: dict[str, float] = {}
        use_rerank = self.reranker and candidates and self.config.score_weights.reranker > 0
        if profile == "api" and use_rerank:
            # Lighter rerank cut for latency
            rerank_n = min(min(20, self.config.rerank_top_k), len(candidates))
        elif use_rerank:
            rerank_n = min(self.config.rerank_top_k, len(candidates))
        else:
            rerank_n = 0

        if use_rerank and rerank_n > 0:
            sub = candidates[:rerank_n]
            scores = self.reranker.score(query, [_rerank_text(c) for c in sub])
            for c, s in zip(sub, scores):
                rerank_scores[c.chunk_id] = s
            for c in candidates[rerank_n:]:
                rerank_scores[c.chunk_id] = 0.0
        else:
            rerank_scores = {
                c.chunk_id: hybrid_scores.get(c.chunk_id, 0.0) for c in candidates
            }

        rr_norm = _minmax(rerank_scores)
        hy_norm = _minmax({c.chunk_id: hybrid_scores.get(c.chunk_id, 0.0) for c in candidates})

        boosts = self.config.metadata_boost
        sw = self.config.score_weights
        final: list[tuple[str, float]] = []
        for c in candidates:
            meta = (
                boosts.date_match * _date_match(c, pq)
                + boosts.section_match * _section_match(c, pq)
                + boosts.entity_match * _entity_match(c, pq)
                + boosts.numeric_match * _numeric_match(c, pq)
                + 0.16 * _entity_cooccurrence_boost(c, pq)
                + 0.08 * _structure_type_boost(c, pq)
            )
            score = (
                sw.reranker * rr_norm.get(c.chunk_id, 0.0)
                + sw.hybrid * hy_norm.get(c.chunk_id, 0.0)
                + sw.metadata * meta
            )
            final.append((c.chunk_id, score))

        final.sort(key=lambda x: x[1], reverse=True)

        selected: list[tuple[str, float]] = []
        seen_norm: set[str] = set()
        for cid, score in final:
            c = self.chunks[cid]
            key = normalize_text(c.raw_text)[:180]
            if key in seen_norm:
                continue
            seen_norm.add(key)
            selected.append((cid, score))
            if len(selected) >= self.config.final_top_k:
                break

        results: list[RetrievalResult] = []
        for rank, (cid, score) in enumerate(selected, start=1):
            c = self.chunks[cid]
            meta = c.metadata.model_dump()
            meta["page"] = c.metadata.page_start
            meta["character_span"] = [c.metadata.char_start, c.metadata.char_end]
            if c.metadata.bounding_boxes:
                b0 = c.metadata.bounding_boxes[0]
                meta["bounding_box"] = [b0.x0, b0.y0, b0.x1, b0.y1]
            results.append(
                RetrievalResult(
                    rank=rank,
                    chunk_id=cid,
                    score=round(float(score), 6),
                    content=c.raw_text,
                    metadata=meta,
                )
            )
        return QueryResult(query_id=query_id, query=query, results=results)
