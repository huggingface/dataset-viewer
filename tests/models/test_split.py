from datasets_preview_backend.config import HF_TOKEN
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
    config_name = "asr"
    split_name = "test"
    split = get_split(dataset_name, config_name, split_name, HF_TOKEN)

    assert len(split["rows"]) == 3
    assert split["rows"][0]["file"] == "https://huggingface.co/datasets/Narsil/asr_dummy/raw/main/1.flac"
