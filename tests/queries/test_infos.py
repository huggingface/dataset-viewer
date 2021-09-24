import pytest

from datasets_preview_backend.config import DATASETS_ENABLE_PRIVATE, HF_TOKEN
from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.infos import get_infos


def test_config() -> None:
    # token is required for the tests
    assert not DATASETS_ENABLE_PRIVATE or HF_TOKEN is not None


def test_get_infos() -> None:
    dataset = "glue"
    config = "ax"
    response = get_infos(dataset, config)
    assert "infos" in response
    infoItems = response["infos"]
    assert len(infoItems) == 1
    infoItem = infoItems[0]
    assert "dataset" in infoItem
    assert infoItem["dataset"] == dataset
    assert "config" in infoItem
    assert infoItem["config"] == config
    assert "info" in infoItem
    info = infoItem["info"]
    assert "features" in info


def test_get_infos_no_config() -> None:
    dataset = "glue"
    response = get_infos(dataset)
    infoItems = response["infos"]
    assert len(infoItems) == 12


def test_script_error() -> None:
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.br-quad-2'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status400Error):
        get_infos("piEsposito/br-quad-2.0")


def test_no_dataset() -> None:
    # the dataset does not exist
    with pytest.raises(Status404Error):
        get_infos("doesnotexist")


def test_no_dataset_no_script() -> None:
    # the dataset does not contain a script
    with pytest.raises(Status404Error):
        get_infos("AConsApart/anime_subtitles_DialoGPT")
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.Test'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status404Error):
        get_infos("TimTreasure4/Test")


def test_hub_private_dataset() -> None:
    if DATASETS_ENABLE_PRIVATE:
        response = get_infos("severo/autonlp-data-imdb-sentiment-analysis", token=HF_TOKEN)
        assert response["infos"] == []

    # TODO: find/create a private dataset with a dataset-info.json file


def test_blocklisted_datasets() -> None:
    # see https://github.com/huggingface/datasets-preview-backend/issues/17
    dataset = "allenai/c4"
    with pytest.raises(Status400Error):
        get_infos(dataset)
