import pytest

from datasets_preview_backend.config import EXTRACT_ROWS_LIMIT
from datasets_preview_backend.constants import DEFAULT_CONFIG_NAME
from datasets_preview_backend.io.cache import cache_directory  # type: ignore
from datasets_preview_backend.models.column.class_label import ClassLabelColumn
from datasets_preview_backend.models.column.default import ColumnType
from datasets_preview_backend.models.config import get_config_names
from datasets_preview_backend.models.info import get_info
from datasets_preview_backend.models.row import get_rows, get_rows_and_columns


def test_cache_directory() -> None:
    # ensure the cache directory is empty, so that this file gets an empty cache
    assert cache_directory is None
    # note that the same cache is used all over this file. We might want to call
    # http://www.grantjenks.com/docs/diskcache/api.html#diskcache.Cache.clear
    # at the beginning of every test to start with an empty cache


# get_rows
def test_get_rows() -> None:
    info = get_info("acronym_identification", DEFAULT_CONFIG_NAME)
    rows, columns = get_rows_and_columns("acronym_identification", DEFAULT_CONFIG_NAME, "train", info)
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert rows[0]["tokens"][0] == "What"


def test_class_label() -> None:
    info = get_info("glue", "cola")
    rows, columns = get_rows_and_columns("glue", "cola", "train", info)
    assert columns[1].type.name == "CLASS_LABEL"
    assert isinstance(columns[1], ClassLabelColumn)
    assert "unacceptable" in columns[1].labels
    assert rows[0]["label"] == 1


def test_mnist() -> None:
    info = get_info("mnist", "mnist")
    rows, columns = get_rows_and_columns("mnist", "mnist", "train", info)
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert rows[0]["image"] == "assets/mnist/___/mnist/train/0/image/image.jpg"


def test_cifar() -> None:
    info = get_info("cifar10", "plain_text")
    rows, columns = get_rows_and_columns("cifar10", "plain_text", "train", info)
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert rows[0]["img"] == "assets/cifar10/___/plain_text/train/0/img/image.jpg"


def test_iter_archive() -> None:
    info = get_info("food101", DEFAULT_CONFIG_NAME)
    rows, columns = get_rows_and_columns("food101", DEFAULT_CONFIG_NAME, "train", info)
    assert len(rows) == EXTRACT_ROWS_LIMIT
    assert rows[0]["image"] == "assets/food101/___/default/train/0/image/2885220.jpg"


def test_dl_1_suffix() -> None:
    # see https://github.com/huggingface/datasets/pull/2843
    rows = get_rows("discovery", "discovery", "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT


def test_txt_zip() -> None:
    # see https://github.com/huggingface/datasets/pull/2856
    rows = get_rows("bianet", "en_to_ku", "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT


def test_pathlib() -> None:
    # see https://github.com/huggingface/datasets/issues/2866
    rows = get_rows("counter", DEFAULT_CONFIG_NAME, "train")
    assert len(rows) == EXTRACT_ROWS_LIMIT


def test_community_with_no_config() -> None:
    config_names = get_config_names(dataset_name="Check/region_1")
    assert config_names == ["default"]
    rows = get_rows("Check/region_1", "default", "train")
    assert len(rows) == 2
    info = get_info("Check/region_1", "default")
    with pytest.raises(KeyError):
        # see https://github.com/huggingface/datasets-preview-backend/issues/78
        get_rows_and_columns("Check/region_1", "default", "train", info)
