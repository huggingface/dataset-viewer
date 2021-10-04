import pytest

from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.splits import get_splits


def test_get_splits() -> None:
    dataset = "acronym_identification"
    config = DEFAULT_CONFIG_NAME
    response = get_splits(dataset=dataset, config=config)
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

    splits = [s["split"] for s in get_splits(dataset="glue", config="ax")["splits"]]
    assert len(splits) == 1
    assert "test" in splits
    assert "train" not in splits

    # uses the fallback to call "builder._split_generators" while https://github.com/huggingface/datasets/issues/2743
    splits = [s["split"] for s in get_splits(dataset="hda_nli_hindi", config="HDA nli hindi")["splits"]]
    assert len(splits) == 3
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    splitItems = get_splits(dataset="classla/copa_hr", config="copa_hr")["splits"]
    assert len(splitItems) == 3

    splitItems = get_splits(dataset="mc4", config="sn")["splits"]
    assert len(splitItems) == 2


def test_get_splits_without_config() -> None:
    dataset = "acronym_identification"
    response1 = get_splits(dataset=dataset)
    response2 = get_splits(dataset=dataset, config=DEFAULT_CONFIG_NAME)
    assert len(response1["splits"]) == 3
    assert response1 == response2

    dataset = "glue"
    response = get_splits(dataset=dataset)
    assert len(response["splits"]) == 34
    assert {"dataset": "glue", "config": "ax", "split": "test"} in response["splits"]

    dataset = "adversarial_qa"
    response = get_splits(dataset=dataset)
    assert len(response["splits"]) == 4 * 3


def test_builder_config_error() -> None:
    with pytest.raises(Status400Error):
        get_splits(dataset="KETI-AIR/nikl", config="spoken.v1.0")
    with pytest.raises(Status400Error):
        get_splits(dataset="nateraw/image-folder", config=DEFAULT_CONFIG_NAME)
    with pytest.raises(Status400Error):
        get_splits(dataset="Valahaar/wsdmt", config=DEFAULT_CONFIG_NAME)


def test_not_found() -> None:
    with pytest.raises(Status404Error):
        get_splits(dataset="doesnotexist", config=DEFAULT_CONFIG_NAME)
    with pytest.raises(Status404Error):
        get_splits(dataset="glue", config="doesnotexist")


def test_blocklisted_datasets() -> None:
    # see https://github.com/huggingface/datasets-preview-backend/issues/17
    dataset = "allenai/c4"
    with pytest.raises(Status400Error):
        get_splits(dataset=dataset)
