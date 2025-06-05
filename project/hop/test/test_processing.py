from types import SimpleNamespace
from hop.processing import clean_answer


def test_clean_alias_list() -> None:
    out = clean_answer("A,B ; C", "sequence gene alias")
    assert out == ["A", "B", "C"]


def test_clean_location_list() -> None:
    pred = SimpleNamespace(answer="17q21.31,13q12.3")
    out = clean_answer(pred, "Disease gene location")
    assert out == ["17q21.31", "13q12.3"]


def test_clean_function() -> None:
    txt = "This gene encodes a protein."
    assert clean_answer(txt, "SNP gene function") == txt.strip()


def test_clean_other() -> None:
    assert clean_answer("  hello  ", "other") == "hello"
