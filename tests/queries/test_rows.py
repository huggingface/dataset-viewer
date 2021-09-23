import pytest

from datasets_preview_backend.config import (
    DATASETS_ENABLE_PRIVATE,
    EXTRACT_ROWS_LIMIT,
    HF_TOKEN,
)
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.rows import get_rows


def test_config() -> None:
    # token is required for the tests
    assert not DATASETS_ENABLE_PRIVATE or HF_TOKEN is not None


def test_get_split_rows() -> None:
    dataset = "acronym_identification"
    config = DEFAULT_CONFIG_NAME
    split = "train"
    response = get_rows(dataset, config, split)
    assert "dataset" in response and response["dataset"] == dataset
    assert "config" in response and response["config"] == config
    assert "split" in response and response["split"] == split
    assert "rows" in response
    rows = response["rows"]
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert rows[0]["tokens"][0] == "What"


def test_get_split_rows_without_config() -> None:
    dataset = "acronym_identification"
    split = "train"
    response1 = get_rows(dataset, None, split)
    response2 = get_rows(dataset, DEFAULT_CONFIG_NAME, split)
    rows = response1["rows"]
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert response1 == response2


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


# TODO: find a private model that works
# def test_hub_private_dataset():
#     if DATASETS_ENABLE_PRIVATE:
#        response = get_rows(
#         "severo/autonlp-data-imdb-sentiment-analysis", "default", "train", token=HF_TOKEN
#       )
#       assert len(response["rows"]) == EXTRACT_ROWS_LIMIT
