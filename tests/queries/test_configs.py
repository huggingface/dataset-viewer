import pytest

from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.configs import get_configs


def test_get_configs() -> None:
    dataset = "acronym_identification"
    response = get_configs(dataset=dataset)
    assert "configs" in response
    configs = response["configs"]
    assert len(configs) == 1
    config = configs[0]
    assert "dataset" in config
    assert config["dataset"] == dataset
    assert "config" in config
    assert config["config"] == DEFAULT_CONFIG_NAME

    configs = get_configs(dataset="glue")["configs"]
    assert len(configs) == 12
    assert {"dataset": "glue", "config": "cola"} in configs


def test_import_nltk() -> None:
    # requires the nltk dependency
    configs = get_configs(dataset="vershasaxena91/squad_multitask")["configs"]
    assert len(configs) == 3


def test_script_error() -> None:
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.br-quad-2'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status400Error):
        get_configs(dataset="piEsposito/br-quad-2.0")


def test_no_dataset() -> None:
    # the dataset does not exist
    with pytest.raises(Status404Error):
        get_configs(dataset="doesnotexist")


def test_no_dataset_no_script() -> None:
    # the dataset does not contain a script
    with pytest.raises(Status404Error):
        get_configs(dataset="AConsApart/anime_subtitles_DialoGPT")
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.Test'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status404Error):
        get_configs(dataset="TimTreasure4/Test")


def test_blocklisted_datasets() -> None:
    # see https://github.com/huggingface/datasets-preview-backend/issues/17
    dataset = "allenai/c4"
    with pytest.raises(Status400Error):
        get_configs(dataset=dataset)
