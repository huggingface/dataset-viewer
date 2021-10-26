import pytest

from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.models.dataset import get_dataset


def test_script_error() -> None:
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.br-quad-2'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status400Error):
        get_dataset(dataset_name="piEsposito/br-quad-2.0")


def test_no_dataset() -> None:
    # the dataset does not exist
    with pytest.raises(Status404Error):
        get_dataset(dataset_name="doesnotexist")


def test_no_dataset_no_script() -> None:
    # the dataset does not contain a script
    with pytest.raises(Status404Error):
        get_dataset(dataset_name="AConsApart/anime_subtitles_DialoGPT")
    # raises "ModuleNotFoundError: No module named 'datasets_modules.datasets.Test'"
    # which should be caught and raised as DatasetBuilderScriptError
    with pytest.raises(Status404Error):
        get_dataset(dataset_name="TimTreasure4/Test")


def test_builder_config_error() -> None:
    with pytest.raises(Status400Error):
        get_dataset(dataset_name="KETI-AIR/nikl")
    with pytest.raises(Status400Error):
        get_dataset(dataset_name="nateraw/image-folder")
    with pytest.raises(Status400Error):
        get_dataset(dataset_name="Valahaar/wsdmt")
