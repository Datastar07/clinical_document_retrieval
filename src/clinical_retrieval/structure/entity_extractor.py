from __future__ import annotations

import re
from typing import Any

from clinical_retrieval.structure.lexicon import lab_terms, load_lexicon, medication_terms
from clinical_retrieval.structure.normalizer import extract_dates


ICD_RE = re.compile(r"\b([A-TV-Z]\d{2}(?:\.\d{1,4})?)\b")
BP_RE = re.compile(r"\b(\d{2,3}\s*/\s*\d{2,3})\s*(?:mmHg)?\b", re.I)
PROVIDER_RE = re.compile(
    r"\b(?:Dr\.\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+,?\s*(?:MD|DO|NP|PA|APRN)\b"
)
SPECIALTY_RE = re.compile(
    r"\b(Cardiology|Ophthalmology|Nephrology|Endocrinology|Podiatry|Neurology|"
    r"Allergy\s*/\s*Immunology|Registered Dietitian|Diabetic Education Program)\b",
    re.I,
)
IMAGING_RE = re.compile(
    r"\b(Renal Ultrasound(?:\s+with\s+Doppler)?|Chest X-ray|Electrocardiogram|ECG|CXR|"
    r"CT(?:\s+scan)?|MRI)\b",
    re.I,
)


def _med_regex(lexicon_path: str | None = None) -> re.Pattern:
    meds = medication_terms(load_lexicon(lexicon_path))
    if not meds:
        meds = [
            "Metformin",
            "Lisinopril",
            "Empagliflozin",
            "Jardiance",
            "Amlodipine",
            "Atorvastatin",
            "Ozempic",
            "Semaglutide",
            "Chlorthalidone",
            "Losartan",
            "Hydrochlorothiazide",
            "Carvedilol",
            "Spironolactone",
        ]
    # Longest first to prefer multi-token matches
    alt = "|".join(re.escape(m) for m in sorted(set(meds), key=len, reverse=True))
    return re.compile(rf"\b({alt})\b", re.I)


def _lab_regex(lexicon_path: str | None = None) -> re.Pattern:
    labs = lab_terms(load_lexicon(lexicon_path)) or ["HbA1c", "LDL", "HDL", "TSH", "BMI", "Weight"]
    alt = "|".join(re.escape(x) for x in sorted(set(labs), key=len, reverse=True))
    return re.compile(
        rf"\b({alt})(?:\s*Cholesterol)?\b[^\d]{{0,20}}"
        rf"(\d+(?:\.\d+)?\s*(?:%|mg/dL|lbs|mmHg)?)",
        re.I,
    )


def extract_entities(text: str, lexicon_path: str | None = None) -> dict[str, Any]:
    med_re = _med_regex(lexicon_path)
    lab_re = _lab_regex(lexicon_path)
    return {
        "dates": extract_dates(text),
        "icd_codes": sorted(set(ICD_RE.findall(text))),
        "medications": sorted({m.group(0) for m in med_re.finditer(text)}, key=str.lower),
        "blood_pressures": sorted({re.sub(r"\s+", "", m.group(1)) for m in BP_RE.finditer(text)}),
        "labs": [f"{m.group(1)} {m.group(2).strip()}" for m in lab_re.finditer(text)],
        "providers": sorted(set(PROVIDER_RE.findall(text))),
        "specialties": sorted({m.group(0) for m in SPECIALTY_RE.finditer(text)}),
        "imaging": sorted({m.group(0) for m in IMAGING_RE.finditer(text)}),
    }
