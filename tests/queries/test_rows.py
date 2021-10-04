import pytest

from datasets_preview_backend.config import (
    EXTRACT_ROWS_LIMIT,
)
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.rows import get_rows


def test_get_split_rows() -> None:
    dataset = "acronym_identification"
    config = DEFAULT_CONFIG_NAME
    split = "train"
    response = get_rows(dataset, config, split)
    assert "rows" in response
    rowItems = response["rows"]
    assert len(rowItems) == EXTRACT_ROWS_LIMIT
    rowItem = rowItems[0]
    assert "dataset" in rowItem and rowItem["dataset"] == dataset
    assert "config" in rowItem and rowItem["config"] == config
    assert "split" in rowItem and rowItem["split"] == split
    assert rowItem["row"]["tokens"][0] == "What"


def test_get_split_features() -> None:
    dataset = "acronym_identification"
    config = DEFAULT_CONFIG_NAME
    split = "train"
    response = get_rows(dataset, config, split)
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


def test_get_split_rows_without_split() -> None:
    dataset = "acronym_identification"
    response = get_rows(dataset, DEFAULT_CONFIG_NAME)
    assert len(response["rows"]) == 3 * EXTRACT_ROWS_LIMIT
    assert len(response["features"]) == 3


def test_get_split_rows_without_config() -> None:
    dataset = "acronym_identification"
    split = "train"
    response1 = get_rows(dataset)
    assert len(response1["rows"]) == 1 * 3 * EXTRACT_ROWS_LIMIT
    assert len(response1["features"]) == 3

    response2 = get_rows(dataset, None, split)
    assert response1 == response2

    dataset = "adversarial_qa"
    response3 = get_rows(dataset)
    assert len(response3["rows"]) == 4 * 3 * EXTRACT_ROWS_LIMIT
    assert len(response3["features"]) == 4 * 6


def test_get_unknown_dataset() -> None:
    with pytest.raises(Status404Error):
        get_rows("doesnotexist", DEFAULT_CONFIG_NAME, "train")
    with pytest.raises(Status404Error):
        get_rows("AConsApart/anime_subtitles_DialoGPT", DEFAULT_CONFIG_NAME, "train")


def test_get_unknown_config() -> None:
    with pytest.raises(Status404Error):
        get_rows("glue", "doesnotexist", "train")
    with pytest.raises(Status404Error):
        get_rows("glue", DEFAULT_CONFIG_NAME, "train")
    with pytest.raises(Status404Error):
        get_rows("TimTreasure4/Test", DEFAULT_CONFIG_NAME, "train")


def test_get_unknown_split() -> None:
    with pytest.raises(Status404Error):
        get_rows("glue", "ax", "train")


def test_get_bogus_config() -> None:
    with pytest.raises(Status400Error):
        get_rows("Valahaar/wsdmt", DEFAULT_CONFIG_NAME, "train")
    with pytest.raises(Status400Error):
        get_rows("nateraw/image-folder", DEFAULT_CONFIG_NAME, "train")


def test_get_not_implemented_split() -> None:
    with pytest.raises(Status400Error):
        get_rows("ade_corpus_v2", "Ade_corpus_v2_classification", "train")


def test_tar_gz_extension() -> None:
    with pytest.raises(Status400Error):
        get_rows("air_dialogue", "air_dialogue_data", "train")


def test_dl_1_suffix() -> None:
    # see https://github.com/huggingface/datasets/pull/2843
    dataset = "discovery"
    config = "discovery"
    split = "train"
    response = get_rows(
        dataset,
        config,
        split,
    )
    rows = response["rows"]
    assert len(rows) == EXTRACT_ROWS_LIMIT


def test_txt_zip() -> None:
    # see https://github.com/huggingface/datasets/pull/2856
    dataset = "bianet"
    config = "en_to_ku"
    split = "train"
    response = get_rows(dataset, config, split)
    rows = response["rows"]
    assert len(rows) == EXTRACT_ROWS_LIMIT


def test_pathlib() -> None:
    # see https://github.com/huggingface/datasets/issues/2866
    response = get_rows("counter", DEFAULT_CONFIG_NAME, "train")
    assert len(response["rows"]) == EXTRACT_ROWS_LIMIT


def test_blocklisted_datasets() -> None:
    # see https://github.com/huggingface/datasets-preview-backend/issues/17
    dataset = "allenai/c4"
    with pytest.raises(Status400Error):
        get_rows(dataset)
