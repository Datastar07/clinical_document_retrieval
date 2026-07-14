from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class BBox(BaseModel):
    page: int
    x0: float
    y0: float
    x1: float
    y1: float


class TextBlock(BaseModel):
    block_id: str
    text: str
    bbox: list[float]
    reading_order: int
    font_size: float | None = None
    is_bold: bool = False
    block_type: Literal["text", "heading", "table", "header", "footer", "other"] = "text"


class PageContent(BaseModel):
    document_id: str
    page_number: int
    width: float
    height: float
    text: str
    clean_text: str
    blocks: list[TextBlock] = Field(default_factory=list)
    char_count: int = 0
    ocr_required: bool = False


class ValidationReport(BaseModel):
    filename: str
    sha256: str
    page_count: int
    is_encrypted: bool
    contains_text: bool
    ocr_required_pages: list[int] = Field(default_factory=list)
    processing_version: str
    timestamp: str
    corrupted_pages: list[int] = Field(default_factory=list)


class EncounterMeta(BaseModel):
    encounter_id: str
    encounter_number: int | None = None
    encounter_type: str | None = None
    encounter_date: str | None = None
    provider: str | None = None
    facility: str | None = None
    start_page: int
    end_page: int


class SectionSpan(BaseModel):
    section: str
    section_heading: str
    encounter_id: str | None = None
    start_page: int
    end_page: int
    text: str
    char_start: int = 0
    char_end: int = 0
    bbox: list[float] | None = None


class ChunkMetadata(BaseModel):
    document_id: str
    source_document: str
    patient_id: str
    page_start: int
    page_end: int
    section: str | None = None
    encounter_id: str | None = None
    encounter_date: str | None = None
    encounter_type: str | None = None
    provider: str | None = None
    facility: str | None = None
    char_start: int = 0
    char_end: int = 0
    bounding_boxes: list[BBox] = Field(default_factory=list)
    chunk_type: str = "section"
    table_type: str | None = None
    previous_chunk_id: str | None = None
    next_chunk_id: str | None = None
    parent_chunk_id: str | None = None
    entities: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    raw_text: str
    retrieval_text: str
    normalized_text: str
    metadata: ChunkMetadata


class RetrievalResult(BaseModel):
    rank: int
    chunk_id: str
    score: float
    content: str
    metadata: dict[str, Any]


class QueryResult(BaseModel):
    query_id: str
    query: str
    results: list[RetrievalResult]


class EvaluationItem(BaseModel):
    query_id: str
    query: str
    expected_pages: str
    ground_truth_evidence: str
    category: str
    source_document: str = ""
