import pytest

from datasets_preview_backend.queries import (
    DatasetBuilderScriptError,
    # DatasetBuilderScriptConfigNoSplitsError,
    ConfigNotFoundError,
    DatasetNotFoundError,
    SplitError,
    SplitNotImplementedError,
    get_dataset_config_names,
    get_config_splits,
    extract_rows,
)


def test_get_configs():
    config_names = get_dataset_config_names("acronym_identification")
    assert len(config_names) == 1
    assert config_names[0] is None

    config_names = get_dataset_config_names("glue")
    assert len(config_names) == 12
    assert "cola" in config_names


def test_get_splits():  # sourcery skip: extract-duplicate-method
    splits = get_config_splits("acronym_identification", None)
    assert len(splits) == 3
    assert "train" in splits

    splits = get_config_splits("glue", "ax")
    assert len(splits) == 1
    assert "test" in splits
    assert "train" not in splits

    # uses the fallback to call "builder._split_generators"
    splits = get_config_splits("hda_nli_hindi", "HDA nli hindi")
    assert len(splits) == 3
    assert "train" in splits
    assert "validation" in splits
    assert "test" in splits

    splits = get_config_splits("classla/copa_hr", "copa_hr")
    assert len(splits) == 3

    splits = get_config_splits("mc4", "sn")
    assert len(splits) == 2


def test_extract_split_rows():
    dataset_id = "acronym_identification"
    config_name = None
    split = "train"
    num_rows = 100
    extract = extract_rows(dataset_id, config_name, split, num_rows)
    assert "dataset_id" in extract and extract["dataset_id"] == dataset_id
    assert "config_name" in extract and extract["config_name"] == config_name
    assert "split" in extract and extract["split"] == split
    assert "rows" in extract
    rows = extract["rows"]
    assert len(rows) == num_rows
    assert rows[0]["tokens"][0] == "What"


def test_extract_split_rows_num_rows():
    dataset_id = "acronym_identification"
    config_name = None
    split = "train"
    num_rows = 20
    extract = extract_rows(dataset_id, config_name, split, num_rows)
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


# def test_extract_bogus_splits():
# not sure if we have an example of such an error
# with pytest.raises(DatasetBuilderScriptConfigNoSplitsError):
#     extract_config_rows("mc4", "sn", 10)


def test_extract_not_implemented_split():
    with pytest.raises(SplitNotImplementedError):
        extract_rows("ade_corpus_v2", "Ade_corpus_v2_classification", "train", 10)


def test_tar_gz_extension():
    with pytest.raises(SplitNotImplementedError):
        extract_rows("air_dialogue", "air_dialogue_data", "train", 10)
