from clinical_retrieval.structure.normalizer import normalize_text, expand_aliases
from clinical_retrieval.structure.table_parser import extract_atomic_facts


def test_normalize_dose():
    assert "10 mg" in normalize_text("10mg daily")


def test_alias_expansion():
    out = expand_aliases("Start Jardiance 10mg")
    assert "empagliflozin" in out


def test_allergy_atomic():
    text = "Allergies\nSubstance\nReaction\nSeverity\nPenicillin\nHives\nModerate\n"
    facts = extract_atomic_facts("allergies", text)
    assert any(f["table_type"] == "allergies" for f in facts)
    assert any("Penicillin" in f["raw_text"] for f in facts)
