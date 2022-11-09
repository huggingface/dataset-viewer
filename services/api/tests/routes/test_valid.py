from http import HTTPStatus
import pytest

from libcache.simple_cache import _clean_cache_database, upsert_response

from api.routes.valid import get_valid, is_valid
from api.utils import CacheKind


@pytest.fixture(autouse=True)
def clean_mongo_databases() -> None:
    _clean_cache_database()


def test_empty() -> None:
    assert get_valid() == []
    assert is_valid("dataset") is False


def test_only_splits() -> None:
    dataset = "dataset"
    upsert_response(kind=CacheKind.SPLITS.value, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    assert get_valid() == []
    assert is_valid("dataset") is False


def test_only_first_rows() -> None:
    dataset = "dataset"
    upsert_response(
        kind=CacheKind.FIRST_ROWS.value,
        dataset=dataset,
        config="config",
        split="split",
        content={},
        http_status=HTTPStatus.OK,
    )
    assert get_valid() == []
    assert is_valid("dataset") is False


def test_splits_and_first_rows_ok() -> None:
    dataset = "dataset"
    upsert_response(kind=CacheKind.SPLITS.value, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    upsert_response(
        kind=CacheKind.FIRST_ROWS.value,
        dataset=dataset,
        config="config",
        split="split",
        content={},
        http_status=HTTPStatus.OK,
    )
    assert get_valid() == [dataset]
    assert is_valid("dataset") is True


def test_splits_and_first_rows_ok_and_error() -> None:
    dataset = "dataset"
    upsert_response(kind=CacheKind.SPLITS.value, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    upsert_response(
        kind=CacheKind.FIRST_ROWS.value,
        dataset=dataset,
        config="config",
        split="split_a",
        content={},
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=CacheKind.FIRST_ROWS.value,
        dataset=dataset,
        config="config",
        split="split_b",
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    assert get_valid() == [dataset]
    assert is_valid("dataset") is True


def test_splits_and_first_rows_only_errors() -> None:
    dataset = "dataset"
    upsert_response(kind=CacheKind.SPLITS.value, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    upsert_response(
        kind=CacheKind.FIRST_ROWS.value,
        dataset=dataset,
        config="config",
        split="split",
        content={},
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )
    assert get_valid() == []
    assert is_valid("dataset") is False


def test_valid_datasets() -> None:
    dataset_a = "dataset_a"
    dataset_b = "dataset_b"
    dataset_c = "dataset_c"
    upsert_response(kind=CacheKind.SPLITS.value, dataset=dataset_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=CacheKind.SPLITS.value, dataset=dataset_b, content={}, http_status=HTTPStatus.OK)
    upsert_response(
        kind=CacheKind.SPLITS.value, dataset=dataset_c, content={}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR
    )
    upsert_response(
        kind=CacheKind.FIRST_ROWS.value,
        dataset=dataset_a,
        config="config",
        split="split",
        content={},
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=CacheKind.FIRST_ROWS.value,
        dataset=dataset_b,
        config="config",
        split="split",
        content={},
        http_status=HTTPStatus.OK,
    )
    assert get_valid() == [dataset_a, dataset_b]
    assert is_valid(dataset_a) is True
    assert is_valid(dataset_b) is True
    assert is_valid(dataset_c) is False
