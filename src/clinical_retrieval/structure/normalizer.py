from __future__ import annotations

import re
from datetime import datetime

from dateutil import parser as date_parser

ALIAS_EXPANSIONS: dict[str, list[str]] = {
    "jardiance": ["empagliflozin", "sglt2 inhibitor"],
    "empagliflozin": ["jardiance", "sglt2 inhibitor"],
    "ozempic": ["semaglutide", "glp-1", "glp1 receptor agonist"],
    "semaglutide": ["ozempic", "glp-1", "glp1 receptor agonist"],
    "htn": ["hypertension"],
    "hypertension": ["htn"],
    "t2dm": ["type 2 diabetes mellitus", "type 2 diabetes"],
    "type 2 diabetes mellitus": ["t2dm", "type 2 diabetes"],
    "hba1c": ["a1c", "hemoglobin a1c"],
    "a1c": ["hba1c", "hemoglobin a1c"],
    "bid": ["twice daily"],
    "twice daily": ["bid"],
    "sc": ["subcutaneous"],
    "subcutaneous": ["sc"],
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    t = text.lower()
    t = t.replace("—", "-").replace("–", "-")
    t = re.sub(r"(\d)\s*mg\b", r"\1 mg", t)
    t = re.sub(r"(\d)mg\b", r"\1 mg", t)
    t = re.sub(r"\bb\.?i\.?d\.?\b", "twice daily", t)
    t = re.sub(r"\bt\.?i\.?d\.?\b", "three times daily", t)
    t = re.sub(r"\bq\.?d\.?\b", "once daily", t)
    t = re.sub(r"\bhba1c\b", "hba1c", t)
    t = re.sub(r"\ba1c\b", "hba1c", t)
    t = normalize_whitespace(t)
    return t


def expand_aliases(text: str) -> str:
    norm = normalize_text(text)
    extras: list[str] = []
    for key, aliases in ALIAS_EXPANSIONS.items():
        if re.search(rf"\b{re.escape(key)}\b", norm):
            extras.extend(aliases)
    if extras:
        return norm + " " + " ".join(sorted(set(extras)))
    return norm


def normalize_date_string(raw: str) -> str | None:
    try:
        return date_parser.parse(raw, fuzzy=True).strftime("%Y-%m-%d")
    except (ValueError, OverflowError, TypeError):
        return None


MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def extract_dates(text: str) -> list[str]:
    dates: list[str] = []
    for m in re.finditer(
        r"\b(?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{1,2},\s+\d{4}\b",
        text,
        re.I,
    ):
        nd = normalize_date_string(m.group(0))
        if nd:
            dates.append(nd)
    for m in re.finditer(r"\b(19|20|21|22|23|24|25|26|27)\d{2}-\d{2}-\d{2}\b", text):
        dates.append(m.group(0))
    # Month + year only
    for m in re.finditer(
        r"\b(January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+(19|20|21|22|23|24|25|26|27)\d{2}\b",
        text,
        re.I,
    ):
        month = MONTHS[m.group(1).lower()]
        year = int(m.group(0).split()[-1])
        dates.append(f"{year:04d}-{month:02d}")
    return sorted(set(dates))
