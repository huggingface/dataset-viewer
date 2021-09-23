import pytest

from datasets_preview_backend.config import DATASETS_ENABLE_PRIVATE, HF_TOKEN
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.splits import get_splits


def test_config() -> None:
    # token is required for the tests
    assert not DATASETS_ENABLE_PRIVATE or HF_TOKEN is not None


def test_get_splits() -> None:
    dataset = "acronym_identification"
    config = DEFAULT_CONFIG_NAME
    response = get_splits(dataset, config)
    assert "splits" in response
    splitItems = response["splits"]
    assert len(splitItems) == 3
    split = splitItems[0]
    assert "dataset" in split
    assert split["dataset"] == dataset
    assert "config" in split
    assert split["config"] == config
    assert "split" in split
    assert split["split"] == "train"

    splits = [s["split"] for s in get_splits("glue", "ax")["splits"]]
    assert len(splits) == 1
    assert "test" in splits
    assert "train" not in splits

    # uses the fallback to call "builder._split_generators" while https://github.com/huggingface/datasets/issues/2743
    splits = [s["split"] for s in get_splits("hda_nli_hindi", "HDA nli hindi")["splits"]]
    assert len(splits) == 3
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    splitItems = get_splits("classla/copa_hr", "copa_hr")["splits"]
    assert len(splitItems) == 3

    splitItems = get_splits("mc4", "sn")["splits"]
    assert len(splitItems) == 2


def test_get_splits_without_config() -> None:
    dataset = "acronym_identification"
    response1 = get_splits(dataset)
    response2 = get_splits(dataset, DEFAULT_CONFIG_NAME)
    assert len(response1["splits"]) == 3
    assert response1 == response2

    dataset = "glue"
    response = get_splits(dataset)
    assert len(response["splits"]) == 34
    assert {"dataset": "glue", "config": "ax", "split": "test"} in response["splits"]

    dataset = "adversarial_qa"
    response = get_splits(dataset)
    assert len(response["splits"]) == 4 * 3


def test_builder_config_error() -> None:
    with pytest.raises(Status400Error):
        get_splits("KETI-AIR/nikl", "spoken.v1.0")
    with pytest.raises(Status400Error):
        get_splits("nateraw/image-folder", DEFAULT_CONFIG_NAME)
    with pytest.raises(Status400Error):
        get_splits("Valahaar/wsdmt", DEFAULT_CONFIG_NAME)


def test_not_found() -> None:
    with pytest.raises(Status404Error):
        get_splits("doesnotexist", DEFAULT_CONFIG_NAME)
    with pytest.raises(Status404Error):
        get_splits("glue", "doesnotexist")


def test_hub_private_dataset() -> None:
    if DATASETS_ENABLE_PRIVATE:
        response = get_splits("severo/autonlp-data-imdb-sentiment-analysis", "default", token=HF_TOKEN)
        assert response["splits"][0]["split"] == "train"
