import pytest

from datasets_preview_backend.exceptions import Status404Error
from datasets_preview_backend.queries.infos import get_infos


def test_get_infos() -> None:
    dataset = "acronym_identification"
    config = "default"
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
    dataset = "acronym_identification"
    response = get_infos(dataset)
    infoItems = response["infos"]
    assert len(infoItems) == 1


def test_get_infos_no_dataset_info_file() -> None:
    dataset = "lhoestq/custom_squad"
    response = get_infos(dataset)
    assert len(response["infos"]) == 1


def test_not_found() -> None:
    with pytest.raises(Status404Error):
        get_infos("doesnotexist")
    with pytest.raises(Status404Error):
        get_infos("acronym_identification", "doesnotexist")
