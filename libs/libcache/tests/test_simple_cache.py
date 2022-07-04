import pytest
from pymongo.errors import DocumentTooLarge

from libcache.simple_cache import (
    DoesNotExist,
    HTTPStatus,
    _clean_database,
    connect_to_cache,
    delete_first_rows_responses,
    delete_splits_responses,
    get_first_rows_response,
    get_first_rows_responses_count_by_status,
    get_splits_response,
    get_splits_responses_count_by_status,
    get_valid_dataset_names,
    mark_first_rows_responses_as_stale,
    mark_splits_responses_as_stale,
    upsert_first_rows_response,
    upsert_splits_response,
)

from ._utils import MONGO_CACHE_DATABASE, MONGO_URL


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_CACHE_DATABASE:
        raise ValueError("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_database()


def test_upsert_splits_response() -> None:
    dataset_name = "test_dataset"
    response = {"splits": [{"dataset_name": dataset_name, "config_name": "test_config", "split_name": "test_split"}]}
    upsert_splits_response(dataset_name, response, HTTPStatus.OK)
    response1, http_status = get_splits_response(dataset_name)
    assert http_status == HTTPStatus.OK
    assert response1 == response

    # ensure it's idempotent
    upsert_splits_response(dataset_name, response, HTTPStatus.OK)
    (response2, _) = get_splits_response(dataset_name)
    assert response2 == response1

    mark_splits_responses_as_stale(dataset_name)
    # we don't have access to the stale field
    # we also don't have access to the updated_at field

    delete_splits_responses(dataset_name)
    with pytest.raises(DoesNotExist):
        get_splits_response(dataset_name)

    mark_splits_responses_as_stale(dataset_name)
    with pytest.raises(DoesNotExist):
        get_splits_response(dataset_name)


def test_upsert_first_rows_response() -> None:
    dataset_name = "test_dataset"
    config_name = "test_config"
    split_name = "test_split"
    response = {"key": "value"}
    upsert_first_rows_response(dataset_name, config_name, split_name, response, HTTPStatus.OK)
    response1, http_status = get_first_rows_response(dataset_name, config_name, split_name)
    assert http_status == HTTPStatus.OK
    assert response1 == response

    # ensure it's idempotent
    upsert_first_rows_response(dataset_name, config_name, split_name, response, HTTPStatus.OK)
    (response2, _) = get_first_rows_response(dataset_name, config_name, split_name)
    assert response2 == response1

    mark_first_rows_responses_as_stale(dataset_name)
    mark_first_rows_responses_as_stale(dataset_name, config_name, split_name)
    # we don't have access to the stale field
    # we also don't have access to the updated_at field

    upsert_first_rows_response(dataset_name, config_name, "test_split2", response, HTTPStatus.OK)
    delete_first_rows_responses(dataset_name, config_name, "test_split2")
    get_first_rows_response(dataset_name, config_name, split_name)

    delete_first_rows_responses(dataset_name)
    with pytest.raises(DoesNotExist):
        get_first_rows_response(dataset_name, config_name, split_name)

    mark_first_rows_responses_as_stale(dataset_name)
    mark_first_rows_responses_as_stale(dataset_name, config_name, split_name)
    with pytest.raises(DoesNotExist):
        get_first_rows_response(dataset_name, config_name, split_name)


def test_big_row() -> None:
    # https://github.com/huggingface/datasets-server/issues/197
    dataset_name = "test_dataset"
    config_name = "test_config"
    split_name = "test_split"
    big_response = {"content": "a" * 100_000_000}
    with pytest.raises(DocumentTooLarge):
        upsert_first_rows_response(dataset_name, config_name, split_name, big_response, HTTPStatus.OK)


def test_valid() -> None:
    assert get_valid_dataset_names() == []

    upsert_splits_response(
        "test_dataset",
        {"key": "value"},
        HTTPStatus.OK,
    )

    assert get_valid_dataset_names() == []

    upsert_first_rows_response(
        "test_dataset",
        "test_config",
        "test_split",
        {
            "key": "value",
        },
        HTTPStatus.OK,
    )

    assert get_valid_dataset_names() == ["test_dataset"]

    upsert_splits_response(
        "test_dataset2",
        {"key": "value"},
        HTTPStatus.OK,
    )

    assert get_valid_dataset_names() == ["test_dataset"]

    upsert_first_rows_response(
        "test_dataset2",
        "test_config2",
        "test_split2",
        {
            "key": "value",
        },
        HTTPStatus.BAD_REQUEST,
    )

    assert get_valid_dataset_names() == ["test_dataset"]

    upsert_first_rows_response(
        "test_dataset2",
        "test_config2",
        "test_split3",
        {
            "key": "value",
        },
        HTTPStatus.OK,
    )

    assert get_valid_dataset_names() == ["test_dataset", "test_dataset2"]


def test_count_by_status() -> None:
    assert get_splits_responses_count_by_status() == {"OK": 0, "BAD_REQUEST": 0, "INTERNAL_SERVER_ERROR": 0}

    upsert_splits_response(
        "test_dataset2",
        {"key": "value"},
        HTTPStatus.OK,
    )

    assert get_splits_responses_count_by_status() == {"OK": 1, "BAD_REQUEST": 0, "INTERNAL_SERVER_ERROR": 0}
    assert get_first_rows_responses_count_by_status() == {"OK": 0, "BAD_REQUEST": 0, "INTERNAL_SERVER_ERROR": 0}

    upsert_first_rows_response(
        "test_dataset",
        "test_config",
        "test_split",
        {
            "key": "value",
        },
        HTTPStatus.OK,
    )

    assert get_first_rows_responses_count_by_status() == {"OK": 1, "BAD_REQUEST": 0, "INTERNAL_SERVER_ERROR": 0}
