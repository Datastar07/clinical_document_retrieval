from clinical_retrieval.schemas import PageContent
from clinical_retrieval.structure.base import get_parser, list_parsers
from clinical_retrieval.structure.parsers import AutoStructureParser, has_soap_markers


def test_parser_registry():
    names = list_parsers()
    assert "synthetic_soap" in names
    assert "generic" in names
    assert "docling" in names
    assert "auto" in names
    assert get_parser("synthetic_soap").name == "synthetic_soap"
    assert get_parser("generic").name == "generic"
    assert get_parser("auto").name == "auto"


def test_generic_parser_fallback_document_encounter():
    pages = [
        PageContent(
            document_id="X",
            page_number=1,
            width=612,
            height=792,
            text="hello",
            clean_text="Demographics page only. No clinical note structure here.",
            blocks=[],
            char_count=40,
        )
    ]
    parser = get_parser("generic")
    enc = parser.parse_encounters(pages)
    assert len(enc) == 1
    assert enc[0].encounter_id == "GEN-DOC"
    secs = parser.parse_sections(pages, enc)
    assert secs
    assert secs[0].section in {"general", "subjective", "plan", "assessment"}


def test_synthetic_parser_still_finds_encounter():
    pages = [
        PageContent(
            document_id="PT-55188",
            page_number=8,
            width=612,
            height=792,
            text="",
            clean_text=(
                "Encounter #3 | ENC-464119 |\n"
                "Annual Physical | February 20, 2024 00:00\n"
                "Provider: Dr. Aarav Ali, MD — Internal Medicine\n"
                "Maple Creek Medical Center\n"
                "S — Subjective\nChief Complaint: Annual physical.\n"
            ),
            blocks=[],
            char_count=100,
        )
    ]
    parser = get_parser("synthetic_soap")
    enc = parser.parse_encounters(pages)
    assert enc[0].encounter_id == "ENC-464119"
    assert enc[0].encounter_date == "2024-02-20"


def test_auto_selects_soap_on_markers():
    pages = [
        PageContent(
            document_id="X",
            page_number=1,
            width=612,
            height=792,
            text="",
            clean_text=(
                "Encounter #1 | ENC-900101\n"
                "S — Subjective\nChief Complaint: check.\n"
                "O — Objective\nVitals normal.\n"
            ),
            blocks=[],
            char_count=80,
        )
    ]
    assert has_soap_markers(pages)
    auto = AutoStructureParser()
    enc = auto.parse_encounters(pages)
    assert auto.selected == "synthetic_soap"
    assert enc and enc[0].encounter_id.startswith("ENC")


def test_auto_selects_generic_on_progress_note():
    pages = [
        PageContent(
            document_id="X",
            page_number=1,
            width=612,
            height=792,
            text="",
            clean_text=(
                "Progress Note\n"
                "Date of Service: June 1, 2028\n"
                "Jane Roe, MD\n"
                "HPI: Diabetes follow-up.\n"
                "Assessment: Stable.\n"
                "Plan: Continue Metformin.\n"
            ),
            blocks=[],
            char_count=100,
        )
    ]
    assert not has_soap_markers(pages)
    auto = AutoStructureParser()
    enc = auto.parse_encounters(pages)
    assert auto.selected == "generic"
    assert enc
    assert not (enc[0].encounter_id or "").startswith("ENC-")


def test_generic_visit_date_segmentation():
    pages = [
        PageContent(
            document_id="X",
            page_number=1,
            width=612,
            height=792,
            text="",
            clean_text="Visit Date: March 3, 2027\nProvider: Alex Kim, MD\nPlan: labs.\n",
            blocks=[],
            char_count=50,
        ),
        PageContent(
            document_id="X",
            page_number=2,
            width=612,
            height=792,
            text="",
            clean_text="Visit Date: April 4, 2027\nProvider: Alex Kim, MD\nPlan: meds.\n",
            blocks=[],
            char_count=50,
        ),
    ]
    parser = get_parser("generic")
    enc = parser.parse_encounters(pages)
    assert len(enc) >= 2
    assert enc[0].encounter_date == "2027-03-03"
