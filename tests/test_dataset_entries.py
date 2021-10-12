import pytest

from datasets_preview_backend.cache import cache_directory  # type: ignore
from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.dataset_entries import (
    get_config_names,
    get_dataset_cache_status,
    get_dataset_entry,
    get_dataset_names,
    get_features,
    get_info,
    get_rows,
    get_split,
    get_split_names,
)
from datasets_preview_backend.exceptions import Status400Error, Status404Error


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert cache_directory is None
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


def test_get_dataset_cache_status_error() -> None:
    report = get_dataset_cache_status("doesnotexist")
    assert report["status"] == "cache_miss"

    # warm the cache
    with pytest.raises(Status404Error):
        get_dataset_entry(dataset="doesnotexist")

    report = get_dataset_cache_status("doesnotexist")
    assert report["status"] == "error"


# get_features
def test_empty_features() -> None:
    configs = get_config_names("allenai/c4")
    info = get_info("allenai/c4", configs[0])
    features = get_features(info)
    assert len(features) == 0


def test_get_features() -> None:
    info = get_info("acronym_identification", DEFAULT_CONFIG_NAME)
    features = get_features(info)
    assert len(features) == 3
    feature = features[0]
    assert "name" in feature
    assert "content" in feature
    assert feature["name"] == "id"
    assert "_type" in feature["content"]
    assert feature["content"]["_type"] == "Value"


# get_rows
def test_get_rows() -> None:
    rows = get_rows("acronym_identification", DEFAULT_CONFIG_NAME, "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert rows[0]["tokens"][0] == "What"


def test_get_not_implemented_split() -> None:
    with pytest.raises(Status400Error):
        get_rows("ade_corpus_v2", "Ade_corpus_v2_classification", "train")


def test_tar_gz_extension() -> None:
    with pytest.raises(Status400Error):
        get_rows("air_dialogue", "air_dialogue_data", "train")


def test_dl_1_suffix() -> None:
    # see https://github.com/huggingface/datasets/pull/2843
    rows = get_rows("discovery", "discovery", "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT


def test_txt_zip() -> None:
    # see https://github.com/huggingface/datasets/pull/2856
    rows = get_rows("bianet", "en_to_ku", "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT


def test_pathlib() -> None:
    # see https://github.com/huggingface/datasets/issues/2866
    rows = get_rows(dataset="counter", config=DEFAULT_CONFIG_NAME, split="train")
    assert len(rows) == EXTRACT_ROWS_LIMIT


# get_split
def test_get_split() -> None:
    split = get_split("glue", "ax", "test")
    assert split["split"] == "test"
    assert "rows" in split


# get_split_names
def test_get_splits_names() -> None:
    dataset = "acronym_identification"
    splits = get_split_names(dataset, DEFAULT_CONFIG_NAME)
    assert len(splits) == 3
    assert "train" in splits


def test_splits_fallback() -> None:
    # uses the fallback to call "builder._split_generators" while https://github.com/huggingface/datasets/issues/2743
    splits = get_split_names("hda_nli_hindi", "HDA nli hindi")
    assert len(splits) == 3
    assert "train" in splits


# get_info
def test_get_info() -> None:
    info = get_info("glue", "ax")
    assert "features" in info


def test_get_info_no_dataset_info_file() -> None:
    info = get_info("lhoestq/custom_squad", "plain_text")
    assert "features" in info


# get_config_names
def test_get_config_names() -> None:
    dataset = "acronym_identification"
    configs = get_config_names(dataset)
    assert len(configs) == 1
    assert configs[0] == DEFAULT_CONFIG_NAME

    configs = get_config_names("glue")
    assert len(configs) == 12
    assert "cola" in configs

    # see https://github.com/huggingface/datasets-preview-backend/issues/17
    configs = get_config_names("allenai/c4")
    assert len(configs) == 1


# get_dataset_entry
def test_script_error() -> None:
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.br-quad-2'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status400Error):
        get_dataset_entry(dataset="piEsposito/br-quad-2.0")


def test_no_dataset() -> None:
    # the dataset does not exist
    with pytest.raises(Status404Error):
        get_dataset_entry(dataset="doesnotexist")


def test_no_dataset_no_script() -> None:
    # the dataset does not contain a script
    with pytest.raises(Status404Error):
        get_dataset_entry(dataset="AConsApart/anime_subtitles_DialoGPT")
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.Test'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status404Error):
        get_dataset_entry(dataset="TimTreasure4/Test")


def test_builder_config_error() -> None:
    with pytest.raises(Status400Error):
        get_dataset_entry(dataset="KETI-AIR/nikl")
    with pytest.raises(Status400Error):
        get_dataset_entry(dataset="nateraw/image-folder")
    with pytest.raises(Status400Error):
        get_dataset_entry(dataset="Valahaar/wsdmt")


# get_dataset_names
def test_get_dataset_names() -> None:
    datasets = get_dataset_names()
    assert len(datasets) > 1000
    assert "glue" in datasets
