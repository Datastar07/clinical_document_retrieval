from clinical_retrieval.ingestion.ocr_fallback import should_ocr_page, tesseract_available


def test_should_ocr_page_threshold():
    assert should_ocr_page("abc", min_text_chars=40) is True
    assert should_ocr_page("x" * 50, min_text_chars=40) is False


def test_tesseract_available_boolean():
    # Environment may or may not have tesseract; just ensure callable returns bool
    assert isinstance(tesseract_available(), bool)
