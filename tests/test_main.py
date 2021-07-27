import pytest

from datasets_preview_backend.main import get_dataset_extract


def test_extract_ok():
    extract = get_dataset_extract("acronym_identification", 100)
    assert len(extract) == 100
    assert extract[0]["tokens"][0] == "What"


def test_extract_num_rows():
    extract = get_dataset_extract("acronym_identification", 20)
    assert len(extract) == 20
    assert extract[0]["tokens"][0] == "What"


def test_extract_unknown_model():
    with pytest.raises(FileNotFoundError):
        extract = get_dataset_extract("doesnotexist", 100)


def test_extract_subset_not_implemented():
    with pytest.raises(ValueError, match="Config name is missing..*"):
        extract = get_dataset_extract("glue", 100)
