from clinical_retrieval.structure.encounter_parser import parse_encounters
from clinical_retrieval.schemas import PageContent


def test_encounter_pattern():
    pages = [
        PageContent(
            document_id="PT-55188",
            page_number=8,
            width=612,
            height=792,
            text="",
            clean_text=(
                "Encounter #3 | ENC-464119 |\n"
                "Annual Physical | February 20, 2024\n"
                "Provider: Dr. Aarav Ali, MD — Internal Medicine\n"
                "Maple Creek Medical Center\n"
            ),
            blocks=[],
            char_count=100,
        ),
        PageContent(
            document_id="PT-55188",
            page_number=11,
            width=612,
            height=792,
            text="",
            clean_text=(
                "Encounter #4 | ENC-383740 |\n"
                "Telehealth Visit | February 15, 2024\n"
            ),
            blocks=[],
            char_count=50,
        ),
    ]
    enc = parse_encounters(pages)
    assert len(enc) == 2
    assert enc[0].encounter_id == "ENC-464119"
    assert enc[0].encounter_type == "Annual Physical"
    assert enc[0].encounter_date == "2024-02-20"
    assert enc[0].provider and "Aarav Ali" in enc[0].provider
    # end_page includes the page where the next encounter starts
    assert enc[0].end_page == 11


def test_encounter_with_trailing_time():
    pages = [
        PageContent(
            document_id="PT-55188",
            page_number=150,
            width=612,
            height=792,
            text="",
            clean_text=(
                "Encounter #59 | ENC-323966 |\n"
                "Office Visit | July 23, 2029 00:00\n"
                "Provider: Dr. Amara Okafor, MD — Family\n"
            ),
            blocks=[],
            char_count=80,
        )
    ]
    enc = parse_encounters(pages)
    assert enc[0].encounter_type == "Office Visit"
    assert enc[0].encounter_date == "2029-07-23"