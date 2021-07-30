import pytest

from datasets_preview_backend.main import (
    DatasetBuilderScriptError,
    DatasetBuilderScriptConfigError,
    DatasetBuilderScriptConfigNoSplitsError,
    ConfigNotFoundError,
    DatasetNotFoundError,
    SplitError,
    SplitNotImplementedError,
    get_dataset_config_names,
    get_config_splits,
    extract_dataset_rows,
    extract_config_rows,
    extract_split_rows,
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


def test_extract_split_rows():
    dataset_id = "acronym_identification"
    config_name = None
    split = "train"
    num_rows = 100
    extract = extract_split_rows(dataset_id, config_name, split, num_rows)
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
    extract = extract_split_rows(dataset_id, config_name, split, num_rows)
    rows = extract["rows"]
    assert len(rows) == 20
    assert rows[0]["tokens"][0] == "What"


def test_extract_unknown_config():
    with pytest.raises(ConfigNameError):
        extract_config_rows("glue", "doesnotexist", 100)
    with pytest.raises(ConfigNameError):
        extract_split_rows("glue", "doesnotexist", "train", 100)


def test_extract_unknown_split():
    with pytest.raises(SplitError):
        extract_split_rows("glue", "ax", "train", 100)


def test_extract_config_rows():
    dataset_id = "glue"
    config_name = "cola"
    num_rows = 100
    extract = extract_config_rows(dataset_id, config_name, num_rows)
    assert "dataset_id" in extract and extract["dataset_id"] == dataset_id
    assert "config_name" in extract and extract["config_name"] == config_name
    assert "splits" in extract
    splits = extract["splits"]
    assert len(splits) == 3
    assert "train" in splits
    split = splits["train"]
    rows = split["rows"]
    assert len(rows) == 100
    assert (
        rows[0]["sentence"]
        == "Our friends won't buy this analysis, let alone the next one we propose."
    )


def test_extract_dataset():
    dataset_id = "acronym_identification"
    num_rows = 100
    extract = extract_dataset_rows(dataset_id, num_rows)
    assert "dataset_id" in extract and extract["dataset_id"] == dataset_id
    assert "configs" in extract
    configs = extract["configs"]
    assert None in configs
    assert len(configs) == 1
    assert len(configs[None]["splits"]["train"]["rows"]) == num_rows

    dataset_id = "adversarial_qa"
    num_rows = 100
    extract = extract_dataset_rows(dataset_id, num_rows)
    configs = extract["configs"]
    assert len(configs) == 4
    assert "adversarialQA" in configs
    assert len(configs["adversarialQA"]["splits"]["train"]["rows"]) == num_rows
    assert configs["adversarialQA"]["splits"]["train"]["rows"][0]["title"] == "Brain"


def test_extract_unknown_dataset():
    with pytest.raises(DatasetNotFoundError):
        extract_dataset_rows("doesnotexist", 100)
    with pytest.raises(DatasetNotFoundError):
        extract_dataset_rows("AConsApart/anime_subtitles_DialoGPT", 100)


def test_extract_unknown_config():
    with pytest.raises(ConfigNotFoundError):
        extract_config_rows("glue", "doesnotexist", 100)


def test_extract_bogus_dataset():
    with pytest.raises(DatasetBuilderScriptError):
        extract_dataset_rows("TimTreasure4/Test", 100)


def test_extract_bogus_config():
    with pytest.raises(DatasetBuilderScriptConfigError):
        extract_config_rows("Valahaar/wsdmt", None, 10)
    with pytest.raises(DatasetBuilderScriptConfigError):
        extract_config_rows("nateraw/image-folder", None, 10)


def test_extract_bogus_splits():
    with pytest.raises(DatasetBuilderScriptConfigNoSplitsError):
        extract_config_rows("hda_nli_hindi", "HDA nli hindi", 10)
    with pytest.raises(DatasetBuilderScriptConfigNoSplitsError):
        extract_config_rows("mc4", "sn", 10)
    with pytest.raises(DatasetBuilderScriptConfigNoSplitsError):
        extract_dataset_rows("classla/copa_hr", 100)
    with pytest.raises(DatasetBuilderScriptConfigNoSplitsError):
        extract_config_rows("classla/copa_hr", "copa_hr", 100)


def test_extract_not_implemented_split():
    with pytest.raises(SplitNotImplementedError):
        extract_split_rows("ade_corpus_v2", "Ade_corpus_v2_classification", "train", 10)
