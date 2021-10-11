import pytest

from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status404Error
from datasets_preview_backend.queries.splits import get_splits


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

    # uses the fallback to call "builder._split_generators" while https://github.com/huggingface/datasets/issues/2743
    splits = [s["split"] for s in get_splits("hda_nli_hindi", "HDA nli hindi")["splits"]]
    assert len(splits) == 3
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits


def test_get_splits_no_config() -> None:
    dataset = "acronym_identification"
    response1 = get_splits(dataset)
    response2 = get_splits(dataset, DEFAULT_CONFIG_NAME)
    assert len(response1["splits"]) == 3
    assert response1 == response2


def test_not_found() -> None:
    with pytest.raises(Status404Error):
        get_splits("doesnotexist")
    with pytest.raises(Status404Error):
        get_splits("acronym_identification", "doesnotexist")
