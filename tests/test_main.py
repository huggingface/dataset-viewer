import pytest

from datasets_preview_backend.main import (
    ConfigNameError,
    get_dataset_extract,
    get_dataset_config_names,
    get_dataset_config_extract,
)


def test_get_configs():
    config_names = get_dataset_config_names("acronym_identification")
    assert len(config_names) == 1
    assert config_names[0] is None

    config_names = get_dataset_config_names("glue")
    assert len(config_names) == 12
    assert "cola" in config_names


def test_extract_without_config():
    dataset_id = "acronym_identification"
    config_name = None
    num_rows = 100
    extract = get_dataset_config_extract(dataset_id, config_name, num_rows)
    assert "dataset_id" in extract and extract["dataset_id"] == dataset_id
    assert "config_name" in extract and extract["config_name"] == config_name
    assert "rows" in extract
    rows = extract["rows"]
    assert len(rows) == num_rows
    assert rows[0]["tokens"][0] == "What"


def test_extract_with_config():
    dataset_id = "glue"
    config_name = "cola"
    num_rows = 100
    extract = get_dataset_config_extract(dataset_id, config_name, num_rows)
    assert "config_name" in extract and extract["config_name"] == config_name
    rows = extract["rows"]
    assert len(rows) == 100
    assert (
        rows[0]["sentence"]
        == "Our friends won't buy this analysis, let alone the next one we propose."
    )


def test_extract_num_rows():
    dataset_id = "acronym_identification"
    config_name = None
    num_rows = 20
    extract = get_dataset_config_extract(dataset_id, config_name, num_rows)
    rows = extract["rows"]
    assert len(rows) == 20
    assert rows[0]["tokens"][0] == "What"


def test_extract_unknown_config():
    with pytest.raises(ConfigNameError):
        get_dataset_config_extract("glue", "doesnotexist", 100)


def test_extract_unknown_split():
    # "aeslc" dataset has no "train" split, while "train" is the hardcoded split used to download
    extract = get_dataset_config_extract("aeslc", None, 100)
    rows = extract["rows"]
    assert len(rows) == 0


def test_extract_dataset_without_config():
    dataset_id = "acronym_identification"
    num_rows = 100
    extract = get_dataset_extract(dataset_id, num_rows)
    assert "dataset_id" in extract and extract["dataset_id"] == dataset_id
    assert "configs" in extract
    configs = extract["configs"]
    assert None in configs
    assert len(configs) == 1
    assert len(configs[None]["rows"]) == num_rows


def test_extract_dataset_with_configs():
    dataset_id = "adversarial_qa"
    num_rows = 100
    extract = get_dataset_extract(dataset_id, num_rows)
    configs = extract["configs"]
    assert len(configs) == 4
    assert "adversarialQA" in configs
    assert len(configs["adversarialQA"]["rows"]) == num_rows
    assert configs["adversarialQA"]["rows"][0]["title"] == "Brain"


def test_extract_unknown_dataset():
    with pytest.raises(FileNotFoundError):
        get_dataset_extract("doesnotexist", 100)
