#!/usr/bin/env python3
"""Generate diverse synthetic clinical PDFs for generalization testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import fitz


def _new_page(doc: fitz.Document, width: float = 612, height: float = 792) -> fitz.Page:
    return doc.new_page(width=width, height=height)


def _write(page: fitz.Page, lines: list[str], *, fontsize: float = 10, y0: float = 48) -> None:
    y = y0
    for line in lines:
        if y > page.rect.height - 48:
            break
        page.insert_text((48, y), line[:110], fontsize=fontsize, fontname="helv")
        y += fontsize + 3


def _save(doc: fitz.Document, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()


def make_soap_variant(out: Path) -> dict:
    doc = fitz.open()
    p = _new_page(doc)
    _write(
        p,
        [
            "Patient: Synth Soap Variant | ID: PT-90001",
            "SYNTHETIC DATA - generalization fixture",
            "",
            "Encounter #1 | ENC-900101",
            "Annual Physical - March 15, 2028",
            "Provider: Dr. Maya Chen, MD - NPI 1111111111",
            "Facility: Oakridge Family Clinic",
            "",
            "S - Subjective",
            "Chief Complaint: Routine annual exam.",
            "HPI: Patient with T2DM and hypertension presents for preventive visit.",
            "",
            "O - Objective / Physical Exam",
            "Vitals: BP 138/86 mmHg | Weight 182 lbs | BMI 29.1",
            "",
            "Laboratory Results",
            "HbA1c 7.4%  High  Ref <6.5%",
            "LDL Cholesterol 118 mg/dL  High",
            "",
            "A - Assessment / Diagnoses",
            "E11.9 - Type 2 diabetes mellitus without complications",
            "I10 - Essential (primary) hypertension",
            "",
            "P - Plan",
            "Continue Metformin 1000 mg BID.",
            "Initiate Jardiance (empagliflozin) 10 mg daily for glycemic control.",
            "Rationale: HbA1c remains above goal at 7.4%.",
            "",
            "Electronically signed by Dr. Maya Chen, MD on March 15, 2028",
        ],
    )
    p2 = _new_page(doc)
    _write(
        p2,
        [
            "Patient: Synth Soap Variant | ID: PT-90001",
            "",
            "Encounter #2 | ENC-900202",
            "Office Visit - July 10, 2028",
            "Provider: Dr. Maya Chen, MD - NPI 1111111111",
            "",
            "S - Subjective",
            "Follow-up of diabetes after SGLT2 initiation.",
            "",
            "Laboratory Results",
            "HbA1c 6.9%  High  Ref <6.5%",
            "",
            "P - Plan",
            "Continue Jardiance 10 mg daily.",
            "Follow-Up Instructions: Return to clinic in 3 months.",
            "",
            "Electronically signed by Dr. Maya Chen, MD on July 10, 2028",
        ],
    )
    pdf = out / "document.pdf"
    _save(doc, pdf)
    evaluation = [
        {
            "query_id": "S001",
            "query": "What diabetes medication was started at the March 2028 annual physical?",
            "category": "medication",
            "ground_truth_evidence": "Initiate Jardiance (empagliflozin) 10 mg daily",
            "expected_pages": "1",
        },
        {
            "query_id": "S002",
            "query": "What was the patient's HbA1c at the March 15, 2028 visit?",
            "category": "lab",
            "ground_truth_evidence": "HbA1c 7.4%",
            "expected_pages": "1",
        },
        {
            "query_id": "S003",
            "query": "Who signed the encounter dated July 10, 2028?",
            "category": "signature",
            "ground_truth_evidence": "Electronically signed by Dr. Maya Chen, MD on July 10, 2028",
            "expected_pages": "2",
        },
        {
            "query_id": "S004",
            "query": "What blood pressure was recorded at the annual physical?",
            "category": "vitals",
            "ground_truth_evidence": "BP 138/86",
            "expected_pages": "1",
        },
        {
            "query_id": "S005",
            "query": "What ICD code was listed for hypertension?",
            "category": "icd",
            "ground_truth_evidence": "I10 - Essential (primary) hypertension",
            "expected_pages": "1",
        },
    ]
    return {
        "id": "soap_variant",
        "expected_parser": "synthetic_soap",
        "pdf": str(pdf),
        "pages": 2,
        "evaluation": evaluation,
    }


def make_progress_note(out: Path) -> dict:
    doc = fitz.open()
    p = _new_page(doc)
    _write(
        p,
        [
            "OUTPATIENT PROGRESS NOTE",
            "Patient Name: Alex Rivera    MRN: MRN-44120",
            "Date of Service: 12-Apr-2030",
            "Visit Type: Follow-Up Visit",
            "Clinician: Priya Nair, MD",
            "Location: Harbor Primary Care",
            "",
            "Chief Complaint",
            "Persistent dyspnea on exertion for 2 weeks.",
            "",
            "History of Present Illness",
            "Patient with known asthma and allergic rhinitis. Reports nighttime cough.",
            "No fever. Using albuterol more frequently.",
            "",
            "Medications",
            "Albuterol MDI 2 puffs PRN",
            "Fluticasone 110 mcg 2 puffs BID",
            "",
            "Assessment",
            "Uncontrolled asthma, likely allergic triggers.",
            "",
            "Plan",
            "Start montelukast 10 mg nightly.",
            "Refer to Allergy / Immunology.",
            "Obtain chest X-ray if symptoms worsen.",
            "",
            "Signed: Priya Nair, MD  12-Apr-2030",
        ],
    )
    p2 = _new_page(doc)
    _write(
        p2,
        [
            "OUTPATIENT PROGRESS NOTE - CONTINUED",
            "Date of Service: 03-May-2030",
            "Clinician: Priya Nair, MD",
            "",
            "Interval History",
            "Improved nighttime symptoms after montelukast.",
            "",
            "Plan",
            "Continue montelukast 10 mg nightly.",
            "Follow up in 8 weeks.",
            "",
            "Signed: Priya Nair, MD  03-May-2030",
        ],
    )
    pdf = out / "document.pdf"
    _save(doc, pdf)
    evaluation = [
        {
            "query_id": "P001",
            "query": "What medication was started during the April 2030 visit?",
            "category": "medication",
            "ground_truth_evidence": "Start montelukast 10 mg nightly",
            "expected_pages": "1",
        },
        {
            "query_id": "P002",
            "query": "Who signed the progress note dated 12-Apr-2030?",
            "category": "signature",
            "ground_truth_evidence": "Signed: Priya Nair, MD  12-Apr-2030",
            "expected_pages": "1",
        },
        {
            "query_id": "P003",
            "query": "What specialty referral was ordered?",
            "category": "referral",
            "ground_truth_evidence": "Refer to Allergy / Immunology",
            "expected_pages": "1",
        },
        {
            "query_id": "P004",
            "query": "What imaging was considered if symptoms worsen?",
            "category": "imaging",
            "ground_truth_evidence": "Obtain chest X-ray if symptoms worsen",
            "expected_pages": "1",
        },
        {
            "query_id": "P005",
            "query": "What was the patient's chief complaint at the April visit?",
            "category": "hpi",
            "ground_truth_evidence": "Persistent dyspnea on exertion for 2 weeks",
            "expected_pages": "1",
        },
    ]
    return {
        "id": "progress_note",
        "expected_parser": "generic",
        "pdf": str(pdf),
        "pages": 2,
        "evaluation": evaluation,
    }


def make_table_heavy(out: Path) -> dict:
    doc = fitz.open()
    p = _new_page(doc)
    _write(
        p,
        [
            "LABORATORY AND MEDICATION FLOW SHEET",
            "Patient: Jordan Lee   ID: PT-77801",
            "Visit Date: 2027-09-01   Provider: Sam Ortiz DO",
            "",
            "Active Medications",
            "Drug                 Dose           Frequency",
            "Lisinopril           20 mg          daily",
            "Atorvastatin         40 mg          nightly",
            "Metformin            500 mg         BID",
            "",
            "Labs",
            "Test                 Value          Flag     Ref",
            "Creatinine           1.1 mg/dL      Normal   0.6-1.3",
            "Potassium            4.2 mmol/L     Normal   3.5-5.1",
            "HbA1c                8.1%           High     <6.5%",
            "TSH                  2.4 mIU/L      Normal   0.4-4.0",
            "",
            "Orders",
            "Increase Metformin to 1000 mg BID due to HbA1c 8.1%.",
            "Repeat HbA1c in 90 days.",
        ],
        fontsize=9,
    )
    p2 = _new_page(doc)
    _write(
        p2,
        [
            "IMAGING / REFERRALS GRID",
            "Visit Date: 2027-09-01",
            "",
            "Imaging Orders",
            "Study                      Indication",
            "Abdominal Ultrasound       Elevated ALT previously",
            "",
            "Referrals",
            "Specialty                  Reason",
            "Endocrinology              Uncontrolled diabetes",
            "",
            "Signature: Sam Ortiz, DO  2027-09-01",
        ],
        fontsize=9,
    )
    pdf = out / "document.pdf"
    _save(doc, pdf)
    evaluation = [
        {
            "query_id": "T001",
            "query": "What was the HbA1c value on the September 2027 flow sheet?",
            "category": "lab",
            "ground_truth_evidence": "HbA1c                8.1%",
            "expected_pages": "1",
        },
        {
            "query_id": "T002",
            "query": "What metformin dose change was ordered?",
            "category": "plan",
            "ground_truth_evidence": "Increase Metformin to 1000 mg BID due to HbA1c 8.1%",
            "expected_pages": "1",
        },
        {
            "query_id": "T003",
            "query": "Which specialty referral was placed?",
            "category": "referral",
            "ground_truth_evidence": "Endocrinology              Uncontrolled diabetes",
            "expected_pages": "2",
        },
        {
            "query_id": "T004",
            "query": "What imaging study was ordered?",
            "category": "imaging",
            "ground_truth_evidence": "Abdominal Ultrasound       Elevated ALT previously",
            "expected_pages": "2",
        },
        {
            "query_id": "T005",
            "query": "What dose of atorvastatin is listed?",
            "category": "medication",
            "ground_truth_evidence": "Atorvastatin         40 mg",
            "expected_pages": "1",
        },
    ]
    return {
        "id": "table_heavy",
        "expected_parser": "generic",
        "pdf": str(pdf),
        "pages": 2,
        "evaluation": evaluation,
    }


def make_no_encounter_ids(out: Path) -> dict:
    doc = fitz.open()
    p = _new_page(doc)
    _write(
        p,
        [
            "CLINICAL SUMMARY",
            "Patient: Casey Morgan  Chart: C-220",
            "",
            "Visit Date: January 8, 2026",
            "Clinician: Jane Roe, MD",
            "Site: Riverbend Medical Group",
            "",
            "Reason for Visit: New antihypertensive therapy discussion.",
            "Vitals: Blood Pressure 162/98 mmHg",
            "",
            "Assessment / Plan",
            "Stage 2 hypertension.",
            "Start amlodipine 5 mg daily.",
            "Lifestyle counseling completed.",
            "",
            "Jane Roe, MD - signed January 8, 2026",
        ],
    )
    p2 = _new_page(doc)
    _write(
        p2,
        [
            "CLINICAL SUMMARY",
            "",
            "Visit Date: February 20, 2026",
            "Clinician: Jane Roe, MD",
            "",
            "Interval History: Home BP readings improved.",
            "Vitals: Blood Pressure 138/84 mmHg",
            "",
            "Plan",
            "Continue amlodipine 5 mg daily.",
            "Add hydrochlorothiazide 12.5 mg daily.",
            "",
            "Jane Roe, MD - signed February 20, 2026",
        ],
    )
    pdf = out / "document.pdf"
    _save(doc, pdf)
    evaluation = [
        {
            "query_id": "N001",
            "query": "What blood pressure medication was started on January 8, 2026?",
            "category": "medication",
            "ground_truth_evidence": "Start amlodipine 5 mg daily",
            "expected_pages": "1",
        },
        {
            "query_id": "N002",
            "query": "What was the blood pressure at the January 2026 visit?",
            "category": "vitals",
            "ground_truth_evidence": "Blood Pressure 162/98 mmHg",
            "expected_pages": "1",
        },
        {
            "query_id": "N003",
            "query": "What medication was added in February 2026?",
            "category": "medication",
            "ground_truth_evidence": "Add hydrochlorothiazide 12.5 mg daily",
            "expected_pages": "2",
        },
        {
            "query_id": "N004",
            "query": "Who signed the February 20, 2026 note?",
            "category": "signature",
            "ground_truth_evidence": "Jane Roe, MD - signed February 20, 2026",
            "expected_pages": "2",
        },
        {
            "query_id": "N005",
            "query": "What clinic site is listed on the January visit?",
            "category": "facility",
            "ground_truth_evidence": "Riverbend Medical Group",
            "expected_pages": "1",
        },
    ]
    return {
        "id": "no_encounter_ids",
        "expected_parser": "generic",
        "pdf": str(pdf),
        "pages": 2,
        "evaluation": evaluation,
    }


def make_scanned_like(out: Path) -> dict:
    doc = fitz.open()
    p = _new_page(doc)
    _write(
        p,
        [
            "SCANNED CHART IMAGE SURROGATE",
            "Patient: Robin Park",
            "Visit Date: 2031-11-05",
            "Provider: Lee Kim MD",
            "Finding: Left knee effusion",
            "Plan: Order MRI left knee",
            "Medication: Start naproxen 500 mg BID with food",
            "Signed Lee Kim MD 2031-11-05",
        ],
        fontsize=9,
        y0=120,
    )
    p2 = _new_page(doc)
    _write(p2, ["Page 2 blank scan", "MRI pending"], fontsize=8, y0=400)
    pdf = out / "document.pdf"
    _save(doc, pdf)
    evaluation = [
        {
            "query_id": "C001",
            "query": "What imaging was ordered for the left knee?",
            "category": "imaging",
            "ground_truth_evidence": "Order MRI left knee",
            "expected_pages": "1",
        },
        {
            "query_id": "C002",
            "query": "What medication was started on 2031-11-05?",
            "category": "medication",
            "ground_truth_evidence": "Start naproxen 500 mg BID with food",
            "expected_pages": "1",
        },
        {
            "query_id": "C003",
            "query": "Who signed the November 2031 note?",
            "category": "signature",
            "ground_truth_evidence": "Signed Lee Kim MD 2031-11-05",
            "expected_pages": "1",
        },
        {
            "query_id": "C004",
            "query": "What finding was documented for the knee?",
            "category": "assessment",
            "ground_truth_evidence": "Left knee effusion",
            "expected_pages": "1",
        },
        {
            "query_id": "C005",
            "query": "What is the patient name on the scanned chart?",
            "category": "demographics",
            "ground_truth_evidence": "Patient: Robin Park",
            "expected_pages": "1",
        },
    ]
    return {
        "id": "scanned_like",
        "expected_parser": "generic",
        "pdf": str(pdf),
        "pages": 2,
        "evaluation": evaluation,
    }


GENERATORS = {
    "soap_variant": make_soap_variant,
    "progress_note": make_progress_note,
    "table_heavy": make_table_heavy,
    "no_encounter_ids": make_no_encounter_ids,
    "scanned_like": make_scanned_like,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synth")
    args = parser.parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    entries = []
    for doc_id, fn in GENERATORS.items():
        entry = fn(root / doc_id)
        eval_path = root / doc_id / "evaluation.json"
        eval_path.write_text(json.dumps(entry["evaluation"], indent=2), encoding="utf-8")
        meta = {k: v for k, v in entry.items() if k != "evaluation"}
        meta["evaluation_path"] = str(eval_path)
        (root / doc_id / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        entries.append(meta)
        print(f"Wrote {doc_id}: {meta['pdf']} ({meta['pages']} pages)")

    manifest = {"documents": entries}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
