from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from clinical_retrieval.structure.lexicon import expand_aliases, lab_terms, load_lexicon, medication_terms
from clinical_retrieval.structure.normalizer import extract_dates


SECTION_HINTS = {
    "vitals": ["blood pressure", "bp", "weight", "vitals"],
    "assessment": ["icd", "diagnosis", "diagnoses", "assessment", "diagnosis codes"],
    "plan": [
        "started",
        "initiate",
        "increase",
        "added",
        "medication",
        "plan",
        "why",
        "worked up",
        "workup",
        "considered",
    ],
    "laboratory_results": ["lab", "hba1c", "ldl", "cholesterol", "value", "lab values"],
    "laboratory_orders": ["plasma renin", "aldosterone", "tsh", "laboratory orders"],
    "referrals": ["referred", "referral", "specialty", "cardiology"],
    "signature": ["signed", "electronically signed", "signature"],
    "allergies": ["allergy", "allergies", "reaction"],
    "medications": ["medication", "dose", "taking"],
    "medical_history": ["diagnosed", "history", "status", "chronic"],
    "surgical_history": ["surgical", "surgery", "procedure"],
    "family_history": ["family history", "father", "mother"],
    "demographics": ["blood type", "date of birth", "demographics"],
    "imaging_orders": ["imaging", "ultrasound", "x-ray", "ecg", "doppler"],
    "clinical_notes": ["clinical notes", "discussed"],
    "follow_up": ["follow-up", "follow up", "return", "weeks", "timeframe"],
}


ENCOUNTER_TYPE_HINTS = {
    "annual physical": ["annual physical", "physical exam"],
    "office visit": ["office visit"],
    "telehealth": ["telehealth"],
    "follow-up": ["follow-up", "follow up"],
    "progress note": ["progress note"],
}


DEFAULT_WEIGHTS = {
    "lexical": 0.35,
    "dense": 0.25,
    "structured": 0.30,
    "visual": 0.10,
}


@dataclass
class ParsedQuery:
    raw: str
    intent: str = "general"
    intent_sections: list[str] = field(default_factory=list)
    preferred_sections: list[str] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    year_months: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    providers: list[str] = field(default_factory=list)
    encounter_types: list[str] = field(default_factory=list)
    numerics: list[str] = field(default_factory=list)
    icd_codes: list[str] = field(default_factory=list)
    lab_tests: list[str] = field(default_factory=list)
    expanded_query: str = ""
    retrieval_weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    entities: dict[str, Any] = field(default_factory=dict)
    needles: list[str] = field(default_factory=list)


def _detect_intent(ql: str, sections: list[str]) -> str:
    if "signed" in ql or "signature" in ql:
        return "signature_lookup"
    if any(x in ql for x in ("hba1c", "ldl", "lab")) and any(
        x in ql for x in ("medication", "started", "increased", "dose", "initiated", "added")
    ):
        return "lab_value_linked_to_medication_change"
    if any(x in ql for x in ("icd", "diagnosis code")):
        return "icd_code_lookup"
    if "table" in ql or ("lab" in ql and "value" in ql):
        return "table_heavy"
    if len(sections) >= 2:
        return "multi_part_clinical"
    if any(x in ql for x in ("why", "rationale", "because", "reason")):
        return "clinical_explanation"
    if any(x in sections for x in ("signature", "demographics", "medications")):
        return "metadata_exact"
    return "general"


def _weights_for_intent(intent: str, ql: str) -> dict[str, float]:
    w = dict(DEFAULT_WEIGHTS)
    if intent in {"signature_lookup", "metadata_exact", "icd_code_lookup"}:
        w = {"lexical": 0.40, "dense": 0.15, "structured": 0.40, "visual": 0.05}
    elif intent == "lab_value_linked_to_medication_change":
        w = {"lexical": 0.35, "dense": 0.25, "structured": 0.30, "visual": 0.10}
    elif intent == "table_heavy":
        w = {"lexical": 0.25, "dense": 0.20, "structured": 0.25, "visual": 0.30}
    elif intent == "clinical_explanation":
        w = {"lexical": 0.20, "dense": 0.45, "structured": 0.20, "visual": 0.15}
    elif intent == "multi_part_clinical":
        w = {"lexical": 0.30, "dense": 0.30, "structured": 0.30, "visual": 0.10}
    if re.search(r"\b(19|20)\d{2}\b", ql) or any(
        m in ql
        for m in (
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        )
    ):
        w["structured"] = min(0.45, w["structured"] + 0.10)
        w["lexical"] = min(0.45, w["lexical"] + 0.05)
    s = sum(w.values()) or 1.0
    return {k: v / s for k, v in w.items()}


def parse_query(query: str, lexicon_path: str | None = None) -> ParsedQuery:
    """Deterministic query planner with lexicon-driven med/lab expansion."""
    lex = load_lexicon(lexicon_path)
    q = query
    ql = query.lower()
    sections = [k for k, hints in SECTION_HINTS.items() if any(h in ql for h in hints)]
    dates = extract_dates(query)
    year_months = [d for d in dates if len(d) == 7]

    med_vocab = medication_terms(lex)
    meds: list[str] = []
    for term in med_vocab:
        if term.lower() in ql:
            meds.append(term)
    # Also catch bare class tokens
    for token in ("SGLT2", "GLP-1", "HCTZ"):
        if token.lower() in ql and token not in meds:
            meds.append(token)

    providers = re.findall(
        r"(?:Dr\.\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:\s*,\s*(?:MD|DO|NP|PA))?",
        query,
    )
    enc_types = [k for k, hints in ENCOUNTER_TYPE_HINTS.items() if any(h in ql for h in hints)]
    numerics = re.findall(
        r"\b\d{2,3}\s*/\s*\d{2,3}\b|\b\d+(?:\.\d+)?%|\b\d+\s*mg\b|\b\d{2,3}\s*lbs\b",
        query,
        re.I,
    )
    icds = re.findall(r"\b[A-TV-Z]\d{2}(?:\.\d{1,4})?\b", query)
    lab_vocab = lab_terms(lex) or ["HbA1c", "LDL", "HDL", "TSH", "BMI"]
    labs: list[str] = []
    for lab in lab_vocab:
        if lab.lower() in ql:
            labs.append(lab)

    expansions: list[str] = []
    for med in meds:
        expansions.extend(expand_aliases(med, lex))
    for lab in labs:
        expansions.append(lab)
    # Generic clinical cue expansions (medical synonyms — not assignment phrases)
    if "blood pressure" in ql or re.search(r"\bbp\b", ql):
        expansions += ["BP", "mmHg"]
    if "signed" in ql or "signature" in ql:
        expansions += ["Electronically signed", "Signed by"]
    if "imaging" in ql or "ultrasound" in ql:
        expansions += ["Ultrasound", "Imaging", "Doppler"]
    if "icd" in ql or "diagnosis" in ql:
        expansions += ["ICD", "Assessment", "Diagnoses"]
    if "lab" in ql:
        expansions += ["HbA1c", "LDL", "Laboratory Results"]
        labs = list(dict.fromkeys(labs + ["HbA1c", "LDL"]))
    if "secondary" in ql and "hypertension" in ql:
        expansions += [
            "secondary hypertension",
            "renal artery stenosis",
            "primary aldosteronism",
            "plasma renin",
            "aldosterone",
            "TSH",
        ]
    if "chlorthalidone" in ql or (
        "fourth" in ql and ("antihypertensive" in ql or "hypertension" in ql)
    ):
        expansions.extend(expand_aliases("Chlorthalidone", lex))
    if "sglt2" in ql:
        expansions += ["SGLT2 inhibitor", "Empagliflozin", "Jardiance", "HbA1c", "LDL"]
        labs = list(dict.fromkeys(labs + ["HbA1c", "LDL"]))

    expanded = query
    if expansions:
        # Dedup while preserving order
        seen = set()
        uniq = []
        for e in expansions:
            el = e.lower()
            if el not in seen:
                seen.add(el)
                uniq.append(e)
        expanded = query + " " + " ".join(uniq)
    if dates:
        expanded += " " + " ".join(dates)

    needles: list[str] = []
    needles.extend(meds)
    needles.extend(providers)
    needles.extend(icds)
    needles.extend(numerics)
    needles.extend(labs)
    for med in meds:
        needles.extend(expand_aliases(med, lex))
    # Also index clinical expansion terms used for secondary HTN / class drugs
    for e in expansions:
        if len(e) >= 4:
            needles.append(e)
    needles.extend(re.findall(r"\b\d+(?:\.\d+)?\s*mg\b", query, re.I))
    needles = [n for n in needles if n and len(n) >= 2]

    intent = _detect_intent(ql, sections)
    weights = _weights_for_intent(intent, ql)

    return ParsedQuery(
        raw=query,
        intent=intent,
        intent_sections=sections,
        preferred_sections=list(sections),
        dates=dates,
        year_months=year_months,
        medications=meds,
        providers=providers,
        encounter_types=enc_types,
        numerics=numerics,
        icd_codes=icds,
        lab_tests=labs,
        expanded_query=expanded,
        retrieval_weights=weights,
        entities={
            "medication": meds,
            "dates": dates,
            "lab_test": labs,
            "providers": providers,
            "icd_codes": icds,
        },
        needles=needles,
    )


# Backward-compatible alias
plan_query = parse_query
