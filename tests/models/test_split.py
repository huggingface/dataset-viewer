from datasets_preview_backend.config import HF_TOKEN, ROWS_MAX_NUMBER
from datasets_preview_backend.models.split import get_split

# TODO: test fallback


def test_get_split() -> None:
    dataset_name = "acronym_identification"
    config_name = "default"
    split_name = "train"
    split = get_split(dataset_name, config_name, split_name)

    assert split["num_bytes"] == 7792803
    assert split["num_examples"] == 14006


def test_gated() -> None:
    dataset_name = "severo/dummy_gated"
    config_name = "severo--embellishments"
    split_name = "train"
    split = get_split(dataset_name, config_name, split_name, HF_TOKEN, rows_max_number=ROWS_MAX_NUMBER)

    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER
    assert split["rows_response"]["rows"][0]["row"]["year"] == "1855"


# TODO: test the truncation
