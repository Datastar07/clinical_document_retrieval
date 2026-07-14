from __future__ import annotations

import re
from difflib import SequenceMatcher

from clinical_retrieval.structure.normalizer import normalize_text


def tokenize_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", normalize_text(text)))


def token_overlap(a: str, b: str) -> float:
    """Fraction of tokens in `a` that also appear in `b`."""
    sa, sb = tokenize_set(a), tokenize_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa)


def fuzzy_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def _contains_normalized(haystack: str, needle: str) -> bool:
    h = normalize_text(haystack)
    n = normalize_text(needle)
    if not n:
        return False
    if n in h:
        return True
    # Allow minor spacing differences around units/slashes
    h2 = h.replace(" ", "")
    n2 = n.replace(" ", "")
    return len(n2) >= 8 and n2 in h2


def evidence_matches(
    chunk_text: str,
    ground_truth: str,
    *,
    token_threshold: float = 0.70,
    fuzzy_threshold: float = 0.82,
) -> bool:
    """Return True if retrieved chunk contains the ground-truth evidence.

    Matching is intentionally strict: page overlap alone is never enough, and
    weak token co-occurrence (e.g. shared 'mg' / 'diabetes') must not count.
    """
    gt = ground_truth.strip()
    if not gt or not chunk_text.strip():
        return False

    # Multi-clause evidence: require every clause (assignment GTs are conjunctive).
    parts = [p.strip() for p in re.split(r";", gt) if p.strip()]
    if len(parts) > 1:
        return all(
            evidence_matches(
                chunk_text,
                p,
                token_threshold=token_threshold,
                fuzzy_threshold=fuzzy_threshold,
            )
            for p in parts
        )

    if _contains_normalized(chunk_text, gt):
        return True

    # Key clinical anchors that should appear for short factual GT strings
    anchors = re.findall(
        r"\b(?:[a-z]\d{2}(?:\.\d+)?)\b|"
        r"\b\d{2,3}/\d{2,3}\b|"
        r"\b\d+(?:\.\d+)?%|"
        r"\b\d+(?:\.\d+)?\s*mg\b|"
        r"\b(?:metformin|penicillin|jardiance|empagliflozin|amlodipine|ozempic|"
        r"semaglutide|chlorthalidone|cardiology|nephrology|appendectomy|"
        r"arthroscopy|hives|a\+|blood type)\b",
        normalize_text(gt),
        re.I,
    )
    if anchors:
        ct = normalize_text(chunk_text)
        missing = []
        for a in anchors:
            an = normalize_text(a)
            if an not in ct and an.replace(" ", "") not in ct.replace(" ", ""):
                missing.append(an)
        # All anchors required for short GT; allow 1 miss only for long GT
        if missing and (len(gt) < 160 or len(missing) > 1):
            return False
        if not missing and token_overlap(gt, chunk_text) >= 0.55:
            return True

    # Soft phrase pieces separated by dashes/commas (still require strong coverage)
    soft_parts = [
        p.strip()
        for p in re.split(r"[—\-]", gt)
        if len(normalize_text(p)) >= 6
    ]
    if len(soft_parts) >= 2:
        soft_hits = sum(1 for p in soft_parts if _contains_normalized(chunk_text, p))
        if soft_hits == len(soft_parts):
            return True
        if soft_hits / len(soft_parts) >= 0.8 and token_overlap(gt, chunk_text) >= 0.65:
            return True

    ov = token_overlap(gt, chunk_text)
    if ov >= token_threshold and len(tokenize_set(gt)) >= 5:
        # Guard against generic clinical boilerplate overlap
        if anchors:
            return True
        # Without anchors, require near-substring fuzzy agreement
        return fuzzy_ratio(gt, chunk_text) >= fuzzy_threshold or ov >= 0.85

    # Atomic chunk contained inside longer GT (e.g. single surgical procedure)
    ct_norm = normalize_text(chunk_text)
    gt_norm = normalize_text(gt)
    if 20 < len(ct_norm) < len(gt_norm) and ct_norm in gt_norm:
        return True

    return False


def parse_expected_pages(raw: str) -> set[int]:
    pages: set[int] = set()
    for part in str(raw).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            pages.update(range(int(a), int(b) + 1))
        else:
            pages.add(int(part))
    return pages
