import pytest

from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.queries.infos import get_infos


def test_get_infos() -> None:
    dataset = "glue"
    config = "ax"
    response = get_infos(dataset=dataset, config=config)
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
    response = get_infos(dataset=dataset)
    infoItems = response["infos"]
    assert len(infoItems) == 12


def test_get_infos_no_dataset_info_file() -> None:
    dataset = "lhoestq/custom_squad"
    response = get_infos(dataset=dataset)
    assert len(response["infos"]) == 1


def test_script_error() -> None:
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.br-quad-2'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status400Error):
        get_infos(dataset="piEsposito/br-quad-2.0")


def test_no_dataset() -> None:
    # the dataset does not exist
    with pytest.raises(Status404Error):
        get_infos(dataset="doesnotexist")


def test_no_dataset_no_script() -> None:
    # the dataset does not contain a script
    with pytest.raises(Status404Error):
        get_infos(dataset="AConsApart/anime_subtitles_DialoGPT")
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.Test'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status404Error):
        get_infos(dataset="TimTreasure4/Test")


def test_blocklisted_datasets() -> None:
    # see https://github.com/huggingface/datasets-preview-backend/issues/17
    dataset = "allenai/c4"
    with pytest.raises(Status400Error):
        get_infos(dataset=dataset)
