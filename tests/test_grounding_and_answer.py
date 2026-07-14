from clinical_retrieval.evaluation.grounding_report import build_grounding_report
from clinical_retrieval.generation.answer import (
    generate_answer,
    prepare_evidence,
    validate_citations,
)
from clinical_retrieval.schemas import QueryResult, RetrievalResult


def test_grounding_report_fields():
    per_query = [
        {
            "query_id": "Q1",
            "hit_at_10": True,
            "page_overlap": True,
            "matched_chunk_id": "c1",
            "expected_pages": [2],
            "top_results": [
                {
                    "chunk_id": "c1",
                    "rank": 1,
                    "score": 0.9,
                    "content": "Blood Type A+",
                    "metadata": {
                        "document_id": "PT-55188",
                        "page": 2,
                        "page_start": 2,
                        "page_end": 2,
                        "section": "Demographics",
                        "character_span": [0, 40],
                        "bounding_box": [1, 2, 3, 4],
                    },
                }
            ],
        }
    ]
    report = build_grounding_report(per_query)
    assert report["summary"]["n_retrieval_hits"] == 1
    assert report["summary"]["page_agreement_among_hits"] == 1.0
    assert report["summary"]["fully_grounded_matched_hits"] == 1.0


def test_extractive_answer_and_citations():
    result = QueryResult(
        query_id="q",
        query="What blood type?",
        results=[
            RetrievalResult(
                rank=1,
                chunk_id="c1",
                score=0.8,
                content="Blood Type: A+",
                metadata={
                    "document_id": "PT-55188",
                    "page": 2,
                    "section": "Demographics",
                    "character_span": [0, 20],
                    "bounding_box": [1, 2, 3, 4],
                },
            )
        ],
    )
    bundle = generate_answer("What blood type?", result, provider="extractive")
    assert bundle.mode == "extractive"
    assert "E1" in bundle.answer
    blocks = prepare_evidence(result)
    used, invalid = validate_citations("Answer [E1] only.", blocks)
    assert used == ["E1"]
    assert invalid == []
