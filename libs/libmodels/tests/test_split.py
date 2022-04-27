from libmodels.config import HF_TOKEN, MAX_SIZE_FALLBACK, ROWS_MAX_NUMBER
from libmodels.split import get_split

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


def test_fallback() -> None:
    # https://github.com/huggingface/datasets/issues/3185
    dataset_name = "samsum"
    config_name = "samsum"
    split_name = "train"
    split = get_split(
        dataset_name,
        config_name,
        split_name,
        HF_TOKEN,
        rows_max_number=ROWS_MAX_NUMBER,
        max_size_fallback=MAX_SIZE_FALLBACK,
    )

    assert len(split["rows_response"]["rows"]) == ROWS_MAX_NUMBER


# TODO: test the truncation
