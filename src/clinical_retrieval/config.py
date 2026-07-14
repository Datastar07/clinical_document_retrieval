from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MetadataBoostConfig(BaseModel):
    date_match: float = 0.12
    section_match: float = 0.08
    entity_match: float = 0.08
    numeric_match: float = 0.10
    encounter_type_match: float = 0.05
    provider_match: float = 0.05


class ScoreWeights(BaseModel):
    reranker: float = 0.65
    hybrid: float = 0.20
    metadata: float = 0.15


class ChannelWeights(BaseModel):
    lexical: float = 0.35
    dense: float = 0.25
    structured: float = 0.30
    visual: float = 0.10


class DocumentConfig(BaseModel):
    source_pdf: str
    document_id: str = "PT-55188"
    patient_id: str = "PT-55188"
    patient_name: str = "Claudette Johnson"
    processing_version: str = "2.0.0"


class PathsConfig(BaseModel):
    processed_dir: str = "data/processed"
    index_dir: str = "data/indexes"
    evaluation_path: str = "data/evaluation/evaluation_dataset.json"
    outputs_dir: str = "outputs"
    page_images_dir: str = "data/processed/page_images"
    visual_mv_dir: str = "data/indexes/visual_mv"
    sqlite_path: str = "data/indexes/clinical_meta.db"
    qdrant_path: str = "/root/clinical_artifacts/qdrant"


class ExtractionConfig(BaseModel):
    header_footer_repeat_threshold: float = 0.6
    min_text_chars_per_page: int = 40
    ocr_enabled: bool = False
    ocr_dpi: int = 200
    ocr_lang: str = "eng"
    ocr_max_pages: int | None = None
    render_pages: bool = True
    render_dpi: int = 144
    render_format: str = "jpeg"
    render_quality: int = 85


class StructureConfig(BaseModel):
    """Select which structure parser to use for encounter/section detection."""

    parser: str = "auto"  # auto | synthetic_soap | generic | docling
    docling_enabled: bool = True
    docling_fallback: str = "synthetic_soap"
    docling_mode: str = "batched"  # inline | batched | skip
    docling_batch_pages: int = 50
    docling_max_pages_inline: int = 200
    lexicon_path: str = "configs/clinical_lexicon.yaml"


class ChunkingConfig(BaseModel):
    section_max_chars: int = 2200
    section_overlap_chars: int = 180
    atomic_max_chars: int = 600
    encounter_summary_max_chars: int = 4500
    create_key_facts: bool = True
    create_page_visual_chunks: bool = True


class RetrievalConfig(BaseModel):
    bm25_top_k: int = 60
    dense_top_k: int = 60
    exact_top_k: int = 80
    structured_top_k: int = 40
    visual_top_k: int = 30
    fusion_top_k: int = 80
    rerank_top_k: int = 50
    final_top_k: int = 10
    rrf_k: int = 60
    enable_bm25: bool = True
    enable_dense: bool = True
    enable_exact: bool = True
    enable_structured: bool = True
    enable_visual: bool = True
    profile: str = "full"  # api | full
    lazy_visual_threshold: float = 0.08
    channel_weights: ChannelWeights = Field(default_factory=ChannelWeights)
    metadata_boost: MetadataBoostConfig = Field(default_factory=MetadataBoostConfig)
    score_weights: ScoreWeights = Field(default_factory=ScoreWeights)


class ModelsConfig(BaseModel):
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_batch_size: int = 32
    # auto | cuda | cpu — resolved at runtime via clinical_retrieval.device
    embedding_device: str = "auto"
    embedding_dim: int = 1024
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    reranker_batch_size: int = 8
    use_reranker: bool = True
    visual_model: str = "vidore/colqwen2.5-v0.2"
    visual_batch_size: int = 1
    visual_enabled: bool = True


class QdrantConfig(BaseModel):
    url: str | None = None
    path: str = "/root/clinical_artifacts/qdrant"
    dense_collection: str = "clinical_dense"
    visual_collection: str = "clinical_visual"
    prefer_grpc: bool = False


class EvaluationConfig(BaseModel):
    top_k: int = 10
    fuzzy_threshold: float = 0.72
    token_overlap_threshold: float = 0.55


class AppConfig(BaseSettings):
    document: DocumentConfig
    paths: PathsConfig = Field(default_factory=PathsConfig)
    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path, "r", encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def resolve(self, root: Path | None = None) -> "AppConfig":
        """Resolve relative paths against project root."""
        root = root or Path.cwd()

        def _abs(p: str) -> str:
            path = Path(p)
            if path.is_absolute():
                return str(path)
            return str((root / path).resolve())

        self.document.source_pdf = _abs(self.document.source_pdf)
        self.paths.processed_dir = _abs(self.paths.processed_dir)
        self.paths.index_dir = _abs(self.paths.index_dir)
        self.paths.evaluation_path = _abs(self.paths.evaluation_path)
        self.paths.outputs_dir = _abs(self.paths.outputs_dir)
        self.paths.page_images_dir = _abs(self.paths.page_images_dir)
        self.paths.visual_mv_dir = _abs(self.paths.visual_mv_dir)
        self.paths.sqlite_path = _abs(self.paths.sqlite_path)
        self.paths.qdrant_path = _abs(self.paths.qdrant_path)
        self.qdrant.path = _abs(self.qdrant.path)
        if self.structure.lexicon_path:
            self.structure.lexicon_path = _abs(self.structure.lexicon_path)
        return self

    def with_device(self, preference: str | None = None, *, log: bool = True) -> "AppConfig":
        """Resolve embedding_device (auto/cuda/cpu) onto this config."""
        from clinical_retrieval.device import apply_device

        return apply_device(self, preference, log=log)
