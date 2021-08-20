import pytest

from datasets_preview_backend.queries.configs import (
    Status400Error,
    Status404Error,
    get_configs,
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


def test_import_nltk():
    # requires the nltk dependency
    configs = get_configs("vershasaxena91/squad_multitask")["configs"]
    assert len(configs) == 3


def test_script_error():
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.br-quad-2'", which should be caught and raised as DatasetBuilderScriptError
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
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.Test'", which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status404Error):
        get_configs("TimTreasure4/Test")


def test_no_dataset_bad_script_name():
    # the dataset script name is incorrect
    with pytest.raises(Status404Error):
        get_configs("Cropinky/rap_lyrics_english")
