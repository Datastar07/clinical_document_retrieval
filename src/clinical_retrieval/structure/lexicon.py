from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


@lru_cache(maxsize=4)
def load_lexicon(path: str | None = None) -> dict[str, Any]:
    candidates = []
    if path:
        candidates.append(Path(path))
    # Common project-relative locations
    here = Path(__file__).resolve()
    root = here.parents[3]  # clinical_system/
    candidates.append(root / "configs" / "clinical_lexicon.yaml")
    cwd = Path.cwd() / "configs" / "clinical_lexicon.yaml"
    candidates.append(cwd)

    for p in candidates:
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            return {
                "medications": [str(x) for x in data.get("medications") or []],
                "aliases": {str(k): [str(a) for a in (v or [])] for k, v in (data.get("aliases") or {}).items()},
                "labs": [str(x) for x in data.get("labs") or []],
                "specialties": [str(x) for x in data.get("specialties") or []],
            }
    return {"medications": [], "aliases": {}, "labs": [], "specialties": []}


def medication_terms(lex: dict[str, Any] | None = None) -> list[str]:
    lex = lex or load_lexicon()
    terms = set(lex.get("medications") or [])
    for brand, generics in (lex.get("aliases") or {}).items():
        terms.add(brand)
        terms.update(generics)
    return sorted(terms, key=len, reverse=True)


def expand_aliases(term: str, lex: dict[str, Any] | None = None) -> list[str]:
    """Return brand/generic aliases for a medication/lab term (including itself)."""
    lex = lex or load_lexicon()
    t = term.strip()
    out = {t}
    aliases = lex.get("aliases") or {}
    # Direct key
    for k, vals in aliases.items():
        if k.lower() == t.lower():
            out.update(vals)
            out.add(k)
        for v in vals:
            if v.lower() == t.lower():
                out.add(k)
                out.update(vals)
    return sorted(out)


def lab_terms(lex: dict[str, Any] | None = None) -> list[str]:
    lex = lex or load_lexicon()
    return list(lex.get("labs") or [])
