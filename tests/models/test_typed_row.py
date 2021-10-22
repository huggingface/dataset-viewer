import pytest

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.io.cache import cache_directory  # type: ignore
from datasets_preview_backend.models.column.class_label import ClassLabelColumn
from datasets_preview_backend.models.column.default import ColumnType
from datasets_preview_backend.models.config import get_config_names
from datasets_preview_backend.models.info import get_info
from datasets_preview_backend.models.typed_row import get_typed_rows_and_columns


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert cache_directory is None
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


def test_detect_types_from_typed_rows() -> None:
    info = get_info("allenai/c4", DEFAULT_CONFIG_NAME)
    typed_rows, columns = get_typed_rows_and_columns("allenai/c4", DEFAULT_CONFIG_NAME, "train", info)
    assert len(typed_rows) == EXTRACT_ROWS_LIMIT
    assert columns[0].type == ColumnType.STRING


def test_class_label() -> None:
    info = get_info("glue", "cola")
    typed_rows, columns = get_typed_rows_and_columns("glue", "cola", "train", info)
    assert columns[1].type.name == "CLASS_LABEL"
    assert isinstance(columns[1], ClassLabelColumn)
    assert "unacceptable" in columns[1].labels
    assert typed_rows[0]["label"] == 1


def test_mnist() -> None:
    info = get_info("mnist", "mnist")
    typed_rows, columns = get_typed_rows_and_columns("mnist", "mnist", "train", info)
    assert len(typed_rows) == EXTRACT_ROWS_LIMIT
    assert typed_rows[0]["image"] == "assets/mnist/___/mnist/train/0/image/image.jpg"
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


def test_cifar() -> None:
    info = get_info("cifar10", "plain_text")
    typed_rows, columns = get_typed_rows_and_columns("cifar10", "plain_text", "train", info)
    assert len(typed_rows) == EXTRACT_ROWS_LIMIT
    assert typed_rows[0]["img"] == "assets/cifar10/___/plain_text/train/0/img/image.jpg"
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


def test_iter_archive() -> None:
    info = get_info("food101", DEFAULT_CONFIG_NAME)
    typed_rows, columns = get_typed_rows_and_columns("food101", DEFAULT_CONFIG_NAME, "train", info)
    assert len(typed_rows) == EXTRACT_ROWS_LIMIT
    assert columns[0].type == ColumnType.RELATIVE_IMAGE_URL


def test_image_url() -> None:
    info = get_info("severo/wit", DEFAULT_CONFIG_NAME)
    typed_rows, columns = get_typed_rows_and_columns("severo/wit", DEFAULT_CONFIG_NAME, "train", info)
    assert len(typed_rows) == EXTRACT_ROWS_LIMIT
    assert columns[2].type == ColumnType.IMAGE_URL


def test_community_with_no_config() -> None:
    config_names = get_config_names(dataset_name="Check/region_1")
    assert config_names == ["default"]
    info = get_info("Check/region_1", DEFAULT_CONFIG_NAME)
    with pytest.raises(KeyError):
        # see https://github.com/huggingface/datasets-preview-backend/issues/78
        typed_rows, columns = get_typed_rows_and_columns("Check/region_1", DEFAULT_CONFIG_NAME, "train", info)
