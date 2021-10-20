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

    assert "columns" in response
    assert len(response["columns"]) == 3
    column_item = response["columns"][0]
    assert "dataset" in column_item
    assert "config" in column_item
    assert "column" in column_item
    column = column_item["column"]
    assert "name" in column
    assert "type" in column
    assert column["name"] == "id"
    assert column["type"] == "STRING"


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
