import pytest

from datasets_preview_backend.queries import (
    DatasetBuilderScriptError,
    DatasetBuilderNoSplitsError,
    ConfigNotFoundError,
    DatasetNotFoundError,
    SplitError,
    SplitNotImplementedError,
    get_configs,
    get_splits,
    extract_rows,
)


def test_get_configs():
    dataset = "acronym_identification"
    response = get_configs(dataset)
    assert "dataset" in response
    assert response["dataset"] == dataset
    assert "configs" in response
    configs = response["configs"]
    assert len(configs) == 1
    assert configs[0] is None

    configs = get_configs("glue")["configs"]
    assert len(configs) == 12
    assert "cola" in configs


def test_get_splits():
    dataset = "acronym_identification"
    config = None
    response = get_splits(dataset, config)
    assert "dataset" in response
    assert response["dataset"] == dataset
    assert "config" in response
    assert response["config"] == config
    assert "splits" in response
    splits = response["splits"]
    assert len(splits) == 3
    assert "train" in splits

    splits = get_splits("glue", "ax")["splits"]
    assert len(splits) == 1
    assert "test" in splits
    assert "train" not in splits

    # uses the fallback to call "builder._split_generators"
    splits = get_splits("hda_nli_hindi", "HDA nli hindi")["splits"]
    assert len(splits) == 3
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    splits = get_splits("classla/copa_hr", "copa_hr")["splits"]
    assert len(splits) == 3

    splits = get_splits("mc4", "sn")["splits"]
    assert len(splits) == 2


def test_extract_split_rows():
    dataset = "acronym_identification"
    config = None
    split = "train"
    num_rows = 100
    extract = extract_rows(dataset, config, split, num_rows)
    assert "dataset" in extract and extract["dataset"] == dataset
    assert "config" in extract and extract["config"] == config
    assert "split" in extract and extract["split"] == split
    assert "rows" in extract
    rows = extract["rows"]
    assert len(rows) == num_rows
    assert rows[0]["tokens"][0] == "What"


def test_extract_split_rows_num_rows():
    dataset = "acronym_identification"
    config = None
    split = "train"
    num_rows = 20
    extract = extract_rows(dataset, config, split, num_rows)
    rows = extract["rows"]
    assert len(rows) == 20
    assert rows[0]["tokens"][0] == "What"


def test_extract_unknown_config():
    with pytest.raises(ConfigNotFoundError):
        extract_rows("glue", "doesnotexist", "train", 100)
    with pytest.raises(ConfigNotFoundError):
        extract_rows("glue", None, "train", 100)


def test_extract_unknown_split():
    with pytest.raises(SplitError):
        extract_rows("glue", "ax", "train", 100)


def test_extract_unknown_dataset():
    with pytest.raises(DatasetNotFoundError):
        extract_rows("doesnotexist", None, "train", 100)
    with pytest.raises(DatasetNotFoundError):
        extract_rows("AConsApart/anime_subtitles_DialoGPT", None, "train", 100)


def test_extract_bogus_dataset():
    with pytest.raises(DatasetBuilderScriptError):
        extract_rows("TimTreasure4/Test", None, "train", 100)


def test_extract_bogus_config():
    with pytest.raises(DatasetBuilderScriptError):
        extract_rows("Valahaar/wsdmt", None, "train", 10)
    with pytest.raises(DatasetBuilderScriptError):
        extract_rows("nateraw/image-folder", None, "train", 10)


def test_extract_bogus_splits():
    # not sure if we have an example of such an error
    with pytest.raises(DatasetBuilderNoSplitsError):
        get_splits("journalists_questions", "plain_text")


def test_extract_not_implemented_split():
    with pytest.raises(SplitNotImplementedError):
        extract_rows("ade_corpus_v2", "Ade_corpus_v2_classification", "train", 10)


def test_tar_gz_extension():
    with pytest.raises(SplitNotImplementedError):
        extract_rows("air_dialogue", "air_dialogue_data", "train", 10)
