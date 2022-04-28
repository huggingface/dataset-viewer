from job_runner.models.column import ColumnType, get_columns
from job_runner.models.column.class_label import ClassLabelColumn
from job_runner.models.info import get_info

# TODO: add a test for each type


def test_class_label() -> None:
    info = get_info("glue", "cola")
    columns = get_columns(info, [])
    assert columns[1].type.name == "CLASS_LABEL"
    assert isinstance(columns[1], ClassLabelColumn)
    assert "unacceptable" in columns[1].labels


def test_empty_features() -> None:
    info = get_info("allenai/c4", "allenai--c4")
    columns = get_columns(info, [])
    assert columns == []


def test_get_columns() -> None:
    info = get_info("acronym_identification", "default")
    columns = get_columns(info, [])
    assert columns is not None and len(columns) == 3
    column = columns[0]
    assert column.name == "id"
    assert column.type == ColumnType.STRING


def test_mnist() -> None:
    info = get_info("mnist", "mnist")
    columns = get_columns(info, [])
    assert columns is not None
    assert columns[0].name == "image"
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


def test_cifar() -> None:
    info = get_info("cifar10", "plain_text")
    columns = get_columns(info, [])
    assert columns is not None
    json = columns[0].as_dict()
    assert json["name"] == "img"
    assert json["type"] == "RELATIVE_IMAGE_URL"


def test_iter_archive() -> None:
    info = get_info("food101", "default")
    columns = get_columns(info, [])
    assert columns is not None
    assert columns[0].name == "image"
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


def test_severo_wit() -> None:
    info = get_info("severo/wit", "default")
    columns = get_columns(info, [])
    assert columns is not None
    assert columns[2].name == "image_url"
    assert columns[2].type == ColumnType.IMAGE_URL


def test_audio() -> None:
    info = get_info("abidlabs/test-audio-1", "test")
    columns = get_columns(info, [])
    assert columns is not None
    assert columns[1].name == "Output"
    assert columns[1].type == ColumnType.AUDIO_RELATIVE_SOURCES
