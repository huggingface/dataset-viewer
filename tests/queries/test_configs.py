import pytest

from datasets_preview_backend.config import DATASETS_ENABLE_PRIVATE, HF_TOKEN
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.configs import get_configs


def test_config() -> None:
    # token is required for the tests
    assert not DATASETS_ENABLE_PRIVATE or HF_TOKEN is not None


def test_get_configs() -> None:
    dataset = "acronym_identification"
    response = get_configs(dataset)
    assert "configs" in response
    configs = response["configs"]
    assert len(configs) == 1
    config = configs[0]
    assert "dataset" in config
    assert config["dataset"] == dataset
    assert "config" in config
    assert config["config"] == DEFAULT_CONFIG_NAME

    configs = get_configs("glue")["configs"]
    assert len(configs) == 12
    assert {"dataset": "glue", "config": "cola"} in configs


def test_import_nltk() -> None:
    # requires the nltk dependency
    configs = get_configs("vershasaxena91/squad_multitask")["configs"]
    assert len(configs) == 3


def test_script_error() -> None:
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.br-quad-2'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status400Error):
        get_configs("piEsposito/br-quad-2.0")


def test_no_dataset() -> None:
    # the dataset does not exist
    with pytest.raises(Status404Error):
        get_configs("doesnotexist")


def test_no_dataset_no_script() -> None:
    # the dataset does not contain a script
    with pytest.raises(Status404Error):
        get_configs("AConsApart/anime_subtitles_DialoGPT")
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.Test'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status404Error):
        get_configs("TimTreasure4/Test")


def test_hub_private_dataset() -> None:
    if DATASETS_ENABLE_PRIVATE:
        response = get_configs("severo/autonlp-data-imdb-sentiment-analysis", token=HF_TOKEN)
        assert response["configs"] == [{"dataset": "severo/autonlp-data-imdb-sentiment-analysis", "config": "default"}]
