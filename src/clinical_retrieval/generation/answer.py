from __future__ import annotations

"""Grounded clinical answer generation over retrieved evidence."""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from clinical_retrieval.schemas import QueryResult


SYSTEM_PROMPT = """You are a clinical document assistant.
Answer ONLY using the numbered evidence blocks provided by the user.
Rules:
1. Cite every factual claim with evidence IDs like [E1] or [E2].
2. If the evidence is insufficient, say so explicitly and do not guess.
3. Do not invent labs, meds, dates, ICD codes, or providers not present in evidence.
4. Prefer concise clinical language. Include page references when helpful.
5. This is decision-support only, not a substitute for clinician judgment.
"""


@dataclass
class EvidenceBlock:
    eid: str
    chunk_id: str
    page: int | None
    section: str | None
    span: list[int] | None
    bbox: list[float] | None
    text: str
    score: float


@dataclass
class AnswerBundle:
    query: str
    provider: str
    model: str
    answer: str
    citations_used: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens_est: int = 0
    grounded: bool = True
    mode: str = "llm"  # llm | extractive


def prepare_evidence(result: QueryResult, max_chars_per_chunk: int = 900) -> list[EvidenceBlock]:
    blocks: list[EvidenceBlock] = []
    for i, r in enumerate(result.results, start=1):
        meta = r.metadata or {}
        span = meta.get("character_span")
        if span is None and meta.get("char_start") is not None:
            span = [meta.get("char_start"), meta.get("char_end")]
        bbox = meta.get("bounding_box")
        if not bbox and meta.get("bounding_boxes"):
            b0 = meta["bounding_boxes"][0]
            if isinstance(b0, dict):
                bbox = [b0.get("x0"), b0.get("y0"), b0.get("x1"), b0.get("y1")]
            else:
                bbox = b0
        text = (r.content or "").strip()
        if len(text) > max_chars_per_chunk:
            text = text[: max_chars_per_chunk - 3] + "..."
        blocks.append(
            EvidenceBlock(
                eid=f"E{i}",
                chunk_id=r.chunk_id,
                page=meta.get("page", meta.get("page_start")),
                section=meta.get("section") or meta.get("section_heading"),
                span=span if isinstance(span, list) else None,
                bbox=bbox if isinstance(bbox, list) else None,
                text=text,
                score=float(r.score),
            )
        )
    return blocks


def build_user_prompt(query: str, blocks: list[EvidenceBlock]) -> str:
    parts = [f"Question: {query}", "", "Evidence:"]
    for b in blocks:
        header = (
            f"[{b.eid}] chunk_id={b.chunk_id} page={b.page} "
            f"section={b.section} score={b.score:.4f}"
        )
        parts.append(header)
        parts.append(b.text)
        parts.append("")
    parts.append(
        "Write the answer with [E#] citations. If evidence is insufficient, say so."
    )
    return "\n".join(parts)


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def validate_citations(answer: str, blocks: list[EvidenceBlock]) -> tuple[list[str], list[str]]:
    valid_ids = {b.eid for b in blocks}
    found = re.findall(r"\[(E\d+)\]", answer)
    used = sorted(set(found), key=lambda x: int(x[1:]))
    invalid = [c for c in used if c not in valid_ids]
    return used, invalid


def extractive_answer(query: str, blocks: list[EvidenceBlock]) -> AnswerBundle:
    """No-LLM fallback: return top evidence snippets as a grounded answer."""
    if not blocks:
        ans = "Insufficient evidence retrieved to answer this question."
        return AnswerBundle(
            query=query,
            provider="none",
            model="extractive",
            answer=ans,
            evidence=[],
            mode="extractive",
            grounded=True,
        )
    lines = [
        f"Extractive answer (no LLM) for: {query}",
        "Top grounded evidence:",
    ]
    cites = []
    for b in blocks[:5]:
        lines.append(f"- [{b.eid}] (page {b.page}, {b.section}): {b.text[:280]}")
        cites.append(b.eid)
    return AnswerBundle(
        query=query,
        provider="none",
        model="extractive",
        answer="\n".join(lines),
        citations_used=cites,
        evidence=[b.__dict__ for b in blocks],
        prompt_tokens_est=0,
        mode="extractive",
        grounded=True,
    )


def call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> str:
    import urllib.error
    import urllib.request

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        url=base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP {exc.code}: {body[:500]}") from exc
    return data["choices"][0]["message"]["content"]


def call_anthropic(
    *,
    api_key: str,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 512,
) -> str:
    import urllib.error
    import urllib.request

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    req = urllib.request.Request(
        url="https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic HTTP {exc.code}: {body[:500]}") from exc
    parts = data.get("content") or []
    texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
    return "\n".join(texts).strip()


def generate_answer(
    query: str,
    result: QueryResult,
    *,
    provider: str = "extractive",
    model: str | None = None,
    max_evidence: int = 6,
    max_chars_per_chunk: int = 900,
    temperature: float = 0.1,
    max_tokens: int = 512,
) -> AnswerBundle:
    """
    provider:
      - extractive: no LLM
      - openai: closed-source OpenAI API
      - anthropic: closed-source Anthropic API
      - ollama: open-source via local OpenAI-compatible API
      - openai_compatible: any OpenAI-compatible endpoint (vLLM, TGI, Groq, etc.)
    """
    trimmed = QueryResult(
        query_id=result.query_id,
        query=result.query,
        results=result.results[:max_evidence],
    )
    blocks = prepare_evidence(trimmed, max_chars_per_chunk=max_chars_per_chunk)
    if not blocks:
        return extractive_answer(query, [])

    provider = (provider or "extractive").lower().strip()
    if provider in {"none", "extractive", "offline"}:
        return extractive_answer(query, blocks)

    user_prompt = build_user_prompt(query, blocks)
    est = _est_tokens(SYSTEM_PROMPT + user_prompt)

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        text = call_openai_compatible(
            base_url=base,
            api_key=api_key,
            model=model,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
        text = call_anthropic(
            api_key=api_key,
            model=model,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=max_tokens,
        )
    elif provider in {"ollama", "openai_compatible", "vllm", "local"}:
        base = os.environ.get(
            "OPENAI_COMPAT_BASE_URL",
            os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1"),
        )
        api_key = os.environ.get("OPENAI_COMPAT_API_KEY", os.environ.get("OLLAMA_API_KEY", "ollama"))
        model = model or os.environ.get("OPENAI_COMPAT_MODEL", os.environ.get("OLLAMA_MODEL", "llama3.1"))
        text = call_openai_compatible(
            base_url=base,
            api_key=api_key,
            model=model,
            system=SYSTEM_PROMPT,
            user=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise ValueError(
            f"Unknown provider '{provider}'. Use extractive|openai|anthropic|ollama|openai_compatible"
        )

    used, invalid = validate_citations(text, blocks)
    if invalid:
        text = text + f"\n\n[Validation] Dropped/unknown citations: {', '.join(invalid)}"

    return AnswerBundle(
        query=query,
        provider=provider,
        model=model or "unknown",
        answer=text,
        citations_used=used,
        invalid_citations=invalid,
        evidence=[b.__dict__ for b in blocks],
        prompt_tokens_est=est,
        grounded=len(invalid) == 0 and len(used) > 0,
        mode="llm",
    )


def bundle_to_dict(bundle: AnswerBundle) -> dict[str, Any]:
    return {
        "query": bundle.query,
        "provider": bundle.provider,
        "model": bundle.model,
        "mode": bundle.mode,
        "answer": bundle.answer,
        "citations_used": bundle.citations_used,
        "invalid_citations": bundle.invalid_citations,
        "grounded": bundle.grounded,
        "prompt_tokens_est": bundle.prompt_tokens_est,
        "evidence": bundle.evidence,
    }
