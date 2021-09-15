import pytest

from datasets_preview_backend.config import HF_TOKEN
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.queries.configs import (
    Status400Error,
    Status404Error,
    get_configs,
)


def test_config():
    # token is required for the tests
    assert HF_TOKEN is not None


def test_get_configs():
    dataset = "acronym_identification"
    response = get_configs(dataset)
    assert "dataset" in response
    assert response["dataset"] == dataset
    assert "configs" in response
    configs = response["configs"]
    assert len(configs) == 1
    assert configs[0] == DEFAULT_CONFIG_NAME

    configs = get_configs("glue")["configs"]
    assert len(configs) == 12
    assert "cola" in configs


def test_missing_argument():
    with pytest.raises(Status400Error):
        get_configs(None)


def test_bad_type_argument():
    with pytest.raises(TypeError):
        get_configs()
    with pytest.raises(TypeError):
        get_configs(1)


def test_import_nltk():
    # requires the nltk dependency
    configs = get_configs("vershasaxena91/squad_multitask")["configs"]
    assert len(configs) == 3


def test_script_error():
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.br-quad-2'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status400Error):
        get_configs("piEsposito/br-quad-2.0")


def test_no_dataset():
    # the dataset does not exist
    with pytest.raises(Status404Error):
        get_configs("doesnotexist")


def test_no_dataset_no_script():
    # the dataset does not contain a script
    with pytest.raises(Status404Error):
        get_configs("AConsApart/anime_subtitles_DialoGPT")
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.Test'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status404Error):
        get_configs("TimTreasure4/Test")


def test_hub_private_dataset():
    response = get_configs("severo/autonlp-data-imdb-sentiment-analysis", use_auth_token=HF_TOKEN)
    assert response["configs"] == ["default"]
