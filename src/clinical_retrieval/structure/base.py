"""Pluggable clinical structure detection.

Register per-source parsers and select them via config:

    structure:
      parser: synthetic_soap   # or generic, docling
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from clinical_retrieval.schemas import EncounterMeta, PageContent, SectionSpan


@runtime_checkable
class StructureParser(Protocol):
    """Contract for source-specific encounter/section detection."""

    name: str

    def parse_encounters(self, pages: list[PageContent]) -> list[EncounterMeta]:
        ...

    def parse_sections(
        self,
        pages: list[PageContent],
        encounters: list[EncounterMeta],
    ) -> list[SectionSpan]:
        ...


_REGISTRY: dict[str, StructureParser] = {}


def register_parser(parser: StructureParser) -> StructureParser:
    _REGISTRY[parser.name] = parser
    return parser


def get_parser(name: str) -> StructureParser:
    # Lazy import so registry is populated
    from clinical_retrieval.structure import parsers as _parsers  # noqa: F401

    key = (name or "synthetic_soap").strip().lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "(none)"
        raise ValueError(f"Unknown structure parser '{name}'. Available: {available}")
    return _REGISTRY[key]


def list_parsers() -> list[str]:
    from clinical_retrieval.structure import parsers as _parsers  # noqa: F401

    return sorted(_REGISTRY)
