from clinical_retrieval.evaluation.evidence_matcher import evidence_matches
from clinical_retrieval.evaluation.metrics import hit_at_k, mrr_at_k


def test_evidence_match_medication():
    chunk = "Initiate Jardiance (empagliflozin) 10mg daily to improve glycemic control given HbA1c of 8.2%"
    gt = "Initiate Jardiance (empagliflozin) 10mg daily to improve glycemic control given HbA1c of 8.2%"
    assert evidence_matches(chunk, gt)


def test_evidence_match_allergy():
    chunk = "Allergies | Penicillin — Reaction: Hives — Severity: Moderate"
    gt = "Penicillin — Reaction: Hives — Severity: Moderate"
    assert evidence_matches(chunk, gt)


def test_metrics():
    ranks = [1, 3, None, 10]
    assert hit_at_k(ranks, 10) == 0.75
    assert mrr_at_k(ranks, 10) == (1 + 1 / 3 + 0 + 0.1) / 4
