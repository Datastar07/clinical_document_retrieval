from __future__ import annotations

import re
from typing import Any


MED_NAMES = [
    "Lisinopril",
    "Metformin",
    "Empagliflozin",
    "Jardiance",
    "Amlodipine",
    "Atorvastatin",
    "Rosuvastatin",
    "Ozempic",
    "Semaglutide",
    "Chlorthalidone",
    "Losartan",
    "Hydrochlorothiazide",
]

ICD_LINE = re.compile(
    r"^(?P<code>[A-Z]\d{2}(?:\.\d{1,4})?)\s*[—\-]\s*(?P<desc>.+)$",
    re.M,
)

PLAN_BULLET = re.compile(r"^[\u2022•\-]\s*(.+)$", re.M)

REFERRAL_LINE = re.compile(
    r"^(?P<specialty>Cardiology|Ophthalmology|Nephrology|Endocrinology|Registered Dietitian|"
    r"Diabetic Education Program|Podiatry|Neurology|Pulmonology|Rheumatology|"
    r"Gastroenterology|Dermatology|Urology|Psychiatry)\b\s*(?P<reason>.+)?$",
    re.I | re.M,
)

BP_PATTERN = re.compile(r"\b(?:BP\b.*?|)(\d{2,3}\s*/\s*\d{2,3})\s*(?:mmHg)?", re.I | re.S)
WEIGHT_PATTERN = re.compile(r"\bWeight\b.*?(\d{2,3})\s*lbs", re.I | re.S)
LAB_ROW = re.compile(
    r"^(?P<test>HbA1c|LDL(?:\s*Cholesterol)?|HDL|Triglycerides|Creatinine|eGFR|TSH|Glucose|"
    r"BMI|Sodium|Potassium|ALT|AST|Hemoglobin)\s*$",
    re.I | re.M,
)

SIGNATURE_PATTERN = re.compile(
    r"Electronically signed by:\s*(?P<provider>Dr\.\s*[^,\n]+(?:,\s*MD)?)\s*"
    r"Date:\s*(?P<date>[A-Za-z]+\s+\d{1,2},\s+\d{4})",
    re.I,
)


def _lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip()]


def _slice_between(lines: list[str], start_pat: str, end_pats: list[str]) -> list[str]:
    start = None
    for i, ln in enumerate(lines):
        if re.search(start_pat, ln, re.I):
            start = i + 1
            break
    if start is None:
        return []
    end = len(lines)
    for i in range(start, len(lines)):
        for ep in end_pats:
            if re.search(ep, lines[i], re.I):
                end = i
                return lines[start:end]
    return lines[start:end]


def extract_demographics_facts(text: str) -> list[dict[str, Any]]:
    facts: list[dict[str, Any]] = []
    lines = _lines(text)

    # Label/value pairs
    labels = {
        "Blood Type": "blood_type",
        "Patient ID": "patient_id",
        "Full Name": "full_name",
        "Date of Birth": "dob",
        "Gender": "gender",
    }
    for i, ln in enumerate(lines[:-1]):
        if ln in labels:
            facts.append(
                {
                    "table_type": "demographics",
                    "columns": {labels[ln]: lines[i + 1]},
                    "search_text": f"{ln}: {lines[i + 1]}",
                    "raw_text": f"{ln}: {lines[i + 1]}",
                }
            )

    # Allergies table: headers then triples
    allergy_block = _slice_between(
        lines,
        r"^Allergies$",
        [r"^Current Medications$", r"^Medical", r"^Social"],
    )
    body = [ln for ln in allergy_block if ln not in {"Substance", "Reaction", "Severity"}]
    for i in range(0, len(body) - 2, 3):
        sub, reaction, severity = body[i], body[i + 1], body[i + 2]
        if severity.lower() not in {"mild", "moderate", "severe"}:
            continue
        facts.append(
            {
                "table_type": "allergies",
                "columns": {"substance": sub, "reaction": reaction, "severity": severity},
                "search_text": (
                    f"Allergies | {sub} — Reaction: {reaction} — Severity: {severity}"
                ),
                "raw_text": f"{sub} — Reaction: {reaction} — Severity: {severity}",
            }
        )

    # Medications: headers then quads
    med_block = _slice_between(
        lines,
        r"^Current Medications$",
        [r"^Medical", r"^Social", r"^Allergies"],
    )
    body = [
        ln
        for ln in med_block
        if ln not in {"Medication", "Dose", "Frequency", "Indication"}
    ]
    for i in range(0, len(body) - 3, 4):
        med, dose, freq, indication = body[i : i + 4]
        facts.append(
            {
                "table_type": "current_medications",
                "columns": {
                    "medication": med,
                    "dose": dose,
                    "frequency": freq,
                    "indication": indication,
                },
                "search_text": (
                    f"Current Medications | {med} | Dose {dose} | Frequency {freq} | "
                    f"Indication {indication}"
                ),
                "raw_text": f"{med} {dose}, {freq}, Indication: {indication}",
            }
        )

    # Medical history triples
    hx_block = _slice_between(
        lines,
        r"^Medical\s*&\s*Surgical History$",
        [r"^Procedure$", r"^Social", r"^Family History"],
    )
    body = [ln for ln in hx_block if ln not in {"Condition", "Year Diagnosed", "Status"}]
    for i in range(0, len(body) - 2, 3):
        cond, year, status = body[i], body[i + 1], body[i + 2]
        if not re.fullmatch(r"\d{4}", year):
            continue
        facts.append(
            {
                "table_type": "medical_history",
                "columns": {"condition": cond, "year": year, "status": status},
                "search_text": (
                    f"Medical History | {cond} — Year Diagnosed {year} — Status: {status}"
                ),
                "raw_text": f"{cond} — Year Diagnosed {year} — Status: {status}",
            }
        )

    # Surgical history
    surg_block = _slice_between(
        lines,
        r"^Procedure$",
        [r"^Social", r"^Family History", r"^Review of Systems"],
    )
    body = [ln for ln in surg_block if ln not in {"Procedure", "Year", "Notes"}]
    # Reconstruct as triples when possible
    i = 0
    while i < len(body) - 2:
        proc, year, notes = body[i], body[i + 1], body[i + 2]
        if re.fullmatch(r"\d{4}", year):
            facts.append(
                {
                    "table_type": "surgical_history",
                    "columns": {"procedure": proc, "year": year, "notes": notes},
                    "search_text": (
                        f"Surgical History | {proc} ({year}, {notes.rstrip('.')})"
                    ),
                    "raw_text": f"{proc} ({year}, {notes.rstrip('.')})",
                }
            )
            i += 3
        else:
            i += 1

    # Family history bullets or lines
    for m in re.finditer(
        r"(?:[•\-]\s*)?(Father|Mother|Sister|Brother):\s*(.+)", text, re.I
    ):
        facts.append(
            {
                "table_type": "family_history",
                "columns": {"relation": m.group(1), "condition": m.group(2).strip()},
                "search_text": f"Family History — {m.group(1)}: {m.group(2).strip()}",
                "raw_text": f"Family History — {m.group(1)}: {m.group(2).strip()}",
            }
        )

    return facts


def extract_atomic_facts(section_key: str, text: str) -> list[dict[str, Any]]:
    """Extract atomic fact dicts from a section for table-row style chunks."""
    facts: list[dict[str, Any]] = []

    if section_key in {
        "demographics",
        "patient_summary",
        "allergies",
        "medications",
        "medical_history",
        "social_family",
        "family_history",
        "general",
    }:
        facts.extend(extract_demographics_facts(text))

    lines = _lines(text)

    if section_key in {"laboratory_results", "objective", "plan"}:
        # Stacked lab table: Test / Value / Reference / Flag repeating
        for i, ln in enumerate(lines):
            if re.fullmatch(
                r"HbA1c|LDL(?:\s*Cholesterol)?|HDL|Triglycerides|Creatinine|eGFR|TSH|Glucose|BMI",
                ln,
                re.I,
            ):
                if i + 1 < len(lines) and re.match(r"^\d", lines[i + 1]):
                    facts.append(
                        {
                            "table_type": "laboratory_results",
                            "columns": {"test": ln, "value": lines[i + 1]},
                            "search_text": f"Laboratory Results | {ln} | Value {lines[i + 1]}",
                            "raw_text": f"{ln}: {lines[i + 1]}",
                        }
                    )

    if section_key in {"assessment"}:
        for m in ICD_LINE.finditer(text):
            d = m.groupdict()
            facts.append(
                {
                    "table_type": "diagnoses",
                    "columns": d,
                    "search_text": f"Assessment / Diagnoses | {d['code']} — {d['desc']}",
                    "raw_text": f"{d['code']} — {d['desc']}",
                }
            )
        # Also group all ICD codes into one chunk for multi-code queries
        codes = ICD_LINE.findall(text)
        if len(codes) >= 2:
            joined = "; ".join(f"{c} — {d}" for c, d in codes)
            facts.append(
                {
                    "table_type": "diagnoses_set",
                    "columns": {"codes": [c for c, _ in codes]},
                    "search_text": f"Assessment / Diagnoses | {joined}",
                    "raw_text": "\n".join(f"{c} — {d}" for c, d in codes),
                }
            )

    if section_key in {"plan"}:
        for m in PLAN_BULLET.finditer(text):
            bullet = m.group(1).strip()
            facts.append(
                {
                    "table_type": "plan_item",
                    "columns": {"item": bullet},
                    "search_text": f"Plan | {bullet}",
                    "raw_text": f"• {bullet}",
                }
            )
        # Keep whole plan as atomic-ish if short bullets missing (prose plans)
        if not PLAN_BULLET.search(text) and len(text) < 2500:
            facts.append(
                {
                    "table_type": "plan_block",
                    "columns": {},
                    "search_text": f"Plan | {text[:1200]}",
                    "raw_text": text,
                }
            )

    if section_key in {"referrals"}:
        # Specialty then reason may be on next lines
        for i, ln in enumerate(lines):
            m = re.match(
                r"^(Cardiology|Ophthalmology|Nephrology|Endocrinology|Registered Dietitian|"
                r"Diabetic Education Program|Podiatry|Neurology)$",
                ln,
                re.I,
            )
            if m:
                reason = lines[i + 1] if i + 1 < len(lines) else ""
                if reason.lower() in {"specialty", "reason"}:
                    continue
                facts.append(
                    {
                        "table_type": "referrals",
                        "columns": {"specialty": m.group(1), "reason": reason},
                        "search_text": f"Referrals | {m.group(1)} — {reason}",
                        "raw_text": f"{m.group(1)} — {reason}",
                    }
                )
        for m in REFERRAL_LINE.finditer(text):
            d = m.groupdict()
            reason = (d.get("reason") or "").strip()
            if not reason:
                continue
            facts.append(
                {
                    "table_type": "referrals",
                    "columns": d,
                    "search_text": f"Referrals | {d['specialty']} — {reason}",
                    "raw_text": f"{d['specialty']} — {reason}",
                }
            )

    if section_key in {"imaging_orders"}:
        for m in re.finditer(
            r"(Renal Ultrasound(?:\s+with\s+Doppler)?|Chest X-ray(?:\s*\(CXR\))?|"
            r"Electrocardiogram(?:\s*\(ECG\))?)",
            text,
            re.I,
        ):
            # grab following reason-ish window
            start = m.end()
            window = text[start : start + 220].replace("\n", " ")
            facts.append(
                {
                    "table_type": "imaging_orders",
                    "columns": {"study": m.group(1), "context": window.strip()},
                    "search_text": f"Imaging Orders | {m.group(1)} — {window.strip()}",
                    "raw_text": f"{m.group(1)} — {window.strip()}",
                }
            )

    if section_key in {
        "signature",
        "follow_up",
        "general",
        "clinical_notes",
        "plan",
        "patient_education",
        "imaging_orders",
        "referrals",
        "laboratory_orders",
    }:
        m = SIGNATURE_PATTERN.search(text)
        if m:
            d = m.groupdict()
            facts.append(
                {
                    "table_type": "signature",
                    "columns": d,
                    "search_text": (
                        f"Electronically signed by: {d['provider']} Date: {d['date']}"
                    ),
                    "raw_text": (
                        f"Electronically signed by: {d['provider']} Date: {d['date']}"
                    ),
                }
            )

    if section_key in {"subjective", "objective", "vitals", "general", "encounter_key_facts"}:
        bps = re.findall(r"\b(\d{2,3}\s*/\s*\d{2,3})\s*(?:mmHg)?\b", text)
        wt = WEIGHT_PATTERN.search(text)
        if bps or wt:
            cols: dict[str, str] = {}
            parts = []
            if bps:
                # Prefer first BP-like reading near vitals header if present
                cols["bp"] = bps[0].replace(" ", "")
                parts.append(f"BP {cols['bp']} mmHg")
            if wt:
                cols["weight"] = f"{wt.group(1)} lbs"
                parts.append(f"Weight {cols['weight']}")
            facts.append(
                {
                    "table_type": "vitals",
                    "columns": cols,
                    "search_text": "Vitals | " + "; ".join(parts),
                    "raw_text": "; ".join(parts),
                }
            )

    if section_key in {"clinical_notes", "follow_up", "plan"}:
        # Keep notable sentences mentioning labs / imaging / follow-up
        for sent in re.split(r"(?<=[.!?])\s+", text):
            if re.search(
                r"HbA1c|follow up|return to the clinic|renal ultrasound|secondary hypertension|"
                r"chlorthalidone|primary aldosteronism",
                sent,
                re.I,
            ):
                s = sent.strip()
                if 40 < len(s) < 600:
                    facts.append(
                        {
                            "table_type": "note_sentence",
                            "columns": {},
                            "search_text": s,
                            "raw_text": s,
                        }
                    )

    return facts
