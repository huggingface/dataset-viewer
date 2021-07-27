from datasets_preview_backend.main import get_dataset_extract


def test_extract_ok():
    extract = get_dataset_extract("acronym_identification", 100)
    assert len(extract) == 100
    assert extract[0]["tokens"][0] == "What"


def test_extract_subset_not_implemented():
    extract = get_dataset_extract("glue", 100)
    assert len(extract) == 0
