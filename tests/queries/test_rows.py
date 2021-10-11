import pytest

from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status404Error
from datasets_preview_backend.queries.rows import get_rows


def test_get_splits() -> None:
    dataset = "acronym_identification"
    config = DEFAULT_CONFIG_NAME
    split = "train"
    response = get_rows(dataset, config, split)
    assert "rows" in response
    rowItems = response["rows"]
    assert len(rowItems) > 3
    rowItem = rowItems[0]
    assert "dataset" in rowItem
    assert rowItem["dataset"] == dataset
    assert "config" in rowItem
    assert rowItem["config"] == config
    assert "split" in rowItem
    assert rowItem["split"] == split
    assert "row" in rowItem
    assert rowItem["row"]["tokens"][0] == "What"

    assert "features" in response
    assert len(response["features"]) == 3
    featureItem = response["features"][0]
    assert "dataset" in featureItem
    assert "config" in featureItem
    assert "feature" in featureItem
    feature = featureItem["feature"]
    assert "name" in feature
    assert "content" in feature
    assert feature["name"] == "id"
    assert "_type" in feature["content"]
    assert feature["content"]["_type"] == "Value"


def test_no_config() -> None:
    dataset = "acronym_identification"
    response1 = get_rows(dataset)
    response2 = get_rows(dataset, DEFAULT_CONFIG_NAME, "train")
    assert len(response1["rows"]) > len(response2["rows"])


def test_no_split() -> None:
    dataset = "acronym_identification"
    response1 = get_rows(dataset, DEFAULT_CONFIG_NAME)
    response2 = get_rows(dataset, DEFAULT_CONFIG_NAME, "train")
    assert len(response1["rows"]) > len(response2["rows"])


def test_not_found() -> None:
    with pytest.raises(Status404Error):
        get_rows("doesnotexist")
    with pytest.raises(Status404Error):
        get_rows("acronym_identification", "doesnotexist")
    with pytest.raises(Status404Error):
        get_rows("acronym_identification", DEFAULT_CONFIG_NAME, "doesnotexist")
