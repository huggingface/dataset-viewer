from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.io.cache import cache_directory  # type: ignore
from datasets_preview_backend.models.column import ColumnType, get_columns_from_info
from datasets_preview_backend.models.column.class_label import ClassLabelColumn
from datasets_preview_backend.models.config import get_config_names
from datasets_preview_backend.models.info import get_info


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert cache_directory is None
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


# TODO: add a test for each type


def test_class_label() -> None:
    info = get_info("glue", "cola")
    columns_or_none = get_columns_from_info(info)
    assert columns_or_none is not None
    assert columns_or_none[1].type.name == "CLASS_LABEL"
    assert isinstance(columns_or_none[1], ClassLabelColumn)
    assert "unacceptable" in columns_or_none[1].labels


def test_empty_features() -> None:
    configs = get_config_names("allenai/c4")
    info = get_info("allenai/c4", configs[0])
    columns = get_columns_from_info(info)
    assert columns is None


def test_get_columns() -> None:
    info = get_info("acronym_identification", DEFAULT_CONFIG_NAME)
    columns = get_columns_from_info(info)
    assert columns is not None and len(columns) == 3
    column = columns[0]
    assert column.name == "id"
    assert column.type == ColumnType.STRING


def test_mnist() -> None:
    info = get_info("mnist", "mnist")
    columns = get_columns_from_info(info)
    assert columns is not None
    assert columns[0].name == "image"
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


def test_cifar() -> None:
    info = get_info("cifar10", "plain_text")
    columns = get_columns_from_info(info)
    assert columns is not None
    json = columns[0].to_json()
    assert json["name"] == "img"
    assert json["type"] == "RELATIVE_IMAGE_URL"


def test_iter_archive() -> None:
    info = get_info("food101", DEFAULT_CONFIG_NAME)
    columns = get_columns_from_info(info)
    assert columns is not None
    assert columns[0].name == "image"
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


def test_severo_wit() -> None:
    info = get_info("severo/wit", DEFAULT_CONFIG_NAME)
    columns = get_columns_from_info(info)
    assert columns is not None
    assert columns[2].name == "image_url"
    assert columns[2].type == ColumnType.IMAGE_URL
