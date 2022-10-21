# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from time import process_time

import pytest
from pymongo.errors import DocumentTooLarge

from libcache.simple_cache import (
    DoesNotExist,
    InvalidCursor,
    InvalidLimit,
    _clean_database,
    delete_first_rows_responses,
    delete_splits_responses,
    get_cache_reports_features,
    get_cache_reports_first_rows,
    get_cache_reports_splits,
    get_datasets_with_some_error,
    get_first_rows_response,
    get_first_rows_responses_count_by_status_and_error_code,
    get_splits_response,
    get_splits_responses_count_by_status_and_error_code,
    get_valid_dataset_names,
    is_dataset_name_valid,
    mark_first_rows_responses_as_stale,
    mark_splits_responses_as_stale,
    upsert_first_rows_response,
    upsert_splits_response,
)


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_database()


def test_upsert_splits_response() -> None:
    dataset_name = "test_dataset"
    response = {"splits": [{"dataset_name": dataset_name, "config_name": "test_config", "split_name": "test_split"}]}
    upsert_splits_response(dataset_name, response, HTTPStatus.OK)
    response1, http_status, error_code = get_splits_response(dataset_name)
    assert http_status == HTTPStatus.OK
    assert response1 == response
    assert error_code is None

    # ensure it's idempotent
    upsert_splits_response(dataset_name, response, HTTPStatus.OK)
    (response2, _, _) = get_splits_response(dataset_name)
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

    upsert_splits_response(dataset_name, response, HTTPStatus.BAD_REQUEST, "error_code")
    response3, http_status, error_code = get_splits_response(dataset_name)
    assert response3 == response
    assert http_status == HTTPStatus.BAD_REQUEST
    assert error_code == "error_code"


def test_upsert_first_rows_response() -> None:
    dataset_name = "test_dataset"
    config_name = "test_config"
    split_name = "test_split"
    response = {"key": "value"}
    upsert_first_rows_response(dataset_name, config_name, split_name, response, HTTPStatus.OK)
    response1, http_status, _ = get_first_rows_response(dataset_name, config_name, split_name)
    assert http_status == HTTPStatus.OK
    assert response1 == response

    # ensure it's idempotent
    upsert_first_rows_response(dataset_name, config_name, split_name, response, HTTPStatus.OK)
    (response2, _, _) = get_first_rows_response(dataset_name, config_name, split_name)
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
    assert get_datasets_with_some_error() == []

    upsert_splits_response(
        "test_dataset",
        {"key": "value"},
        HTTPStatus.OK,
    )

    assert get_valid_dataset_names() == []
    assert get_datasets_with_some_error() == []
    assert is_dataset_name_valid("test_dataset") is False
    assert is_dataset_name_valid("test_dataset2") is False
    assert is_dataset_name_valid("test_dataset3") is False

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
    assert get_datasets_with_some_error() == []
    assert is_dataset_name_valid("test_dataset") is True
    assert is_dataset_name_valid("test_dataset2") is False
    assert is_dataset_name_valid("test_dataset3") is False

    upsert_splits_response(
        "test_dataset2",
        {"key": "value"},
        HTTPStatus.OK,
    )

    assert get_valid_dataset_names() == ["test_dataset"]
    assert get_datasets_with_some_error() == []
    assert is_dataset_name_valid("test_dataset") is True
    assert is_dataset_name_valid("test_dataset2") is False
    assert is_dataset_name_valid("test_dataset3") is False

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
    assert get_datasets_with_some_error() == ["test_dataset2"]
    assert is_dataset_name_valid("test_dataset") is True
    assert is_dataset_name_valid("test_dataset2") is False
    assert is_dataset_name_valid("test_dataset3") is False

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
    assert get_datasets_with_some_error() == ["test_dataset2"]
    assert is_dataset_name_valid("test_dataset") is True
    assert is_dataset_name_valid("test_dataset2") is True
    assert is_dataset_name_valid("test_dataset3") is False

    upsert_splits_response(
        "test_dataset3",
        {"key": "value"},
        HTTPStatus.BAD_REQUEST,
    )

    assert get_valid_dataset_names() == ["test_dataset", "test_dataset2"]
    assert get_datasets_with_some_error() == ["test_dataset2", "test_dataset3"]
    assert is_dataset_name_valid("test_dataset") is True
    assert is_dataset_name_valid("test_dataset2") is True
    assert is_dataset_name_valid("test_dataset3") is False


def test_count_by_status_and_error_code() -> None:
    assert "OK" not in get_splits_responses_count_by_status_and_error_code()

    upsert_splits_response(
        "test_dataset",
        {"key": "value"},
        HTTPStatus.OK,
    )

    assert get_splits_responses_count_by_status_and_error_code() == {"200": {None: 1}}
    assert get_first_rows_responses_count_by_status_and_error_code() == {}

    upsert_first_rows_response(
        "test_dataset",
        "test_config",
        "test_split",
        {
            "key": "value",
        },
        HTTPStatus.OK,
    )

    assert get_first_rows_responses_count_by_status_and_error_code() == {"200": {None: 1}}

    upsert_first_rows_response(
        "test_dataset",
        "test_config",
        "test_split2",
        {
            "key": "value",
        },
        HTTPStatus.INTERNAL_SERVER_ERROR,
        error_code="error_code",
    )

    assert get_first_rows_responses_count_by_status_and_error_code() == {
        "200": {None: 1},
        "500": {"error_code": 1},
    }


def test_get_cache_reports_splits() -> None:
    assert get_cache_reports_splits("", 2) == {"cache_reports": [], "next_cursor": ""}
    upsert_splits_response(
        "a",
        {"key": "value"},
        HTTPStatus.OK,
    )
    b_details = {
        "error": "error B",
        "cause_exception": "ExceptionB",
        "cause_message": "Cause message B",
        "cause_traceback": ["B"],
    }
    upsert_splits_response(
        "b",
        b_details,
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "ErrorCodeB",
        b_details,
    )
    c_details = {
        "error": "error C",
        "cause_exception": "ExceptionC",
        "cause_message": "Cause message C",
        "cause_traceback": ["C"],
    }
    upsert_splits_response(
        "c",
        {
            "error": c_details["error"],
        },
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "ErrorCodeC",
        c_details,
    )
    response = get_cache_reports_splits("", 2)
    assert response["cache_reports"] == [
        {"dataset": "a", "http_status": HTTPStatus.OK.value, "error_code": None},
        {
            "dataset": "b",
            "http_status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "error_code": "ErrorCodeB",
        },
    ]
    assert response["next_cursor"] != ""
    next_cursor = response["next_cursor"]

    response = get_cache_reports_splits(next_cursor, 2)
    assert response == {
        "cache_reports": [
            {
                "dataset": "c",
                "http_status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                "error_code": "ErrorCodeC",
            },
        ],
        "next_cursor": "",
    }

    with pytest.raises(InvalidCursor):
        get_cache_reports_splits("not an objectid", 2)
    with pytest.raises(InvalidLimit):
        get_cache_reports_splits(next_cursor, -1)
    with pytest.raises(InvalidLimit):
        get_cache_reports_splits(next_cursor, 0)


def test_get_cache_reports_first_rows() -> None:
    assert get_cache_reports_first_rows("", 2) == {"cache_reports": [], "next_cursor": ""}
    upsert_first_rows_response(
        "a",
        "config",
        "split",
        {"key": "value"},
        HTTPStatus.OK,
    )
    b_details = {
        "error": "error B",
        "cause_exception": "ExceptionB",
        "cause_message": "Cause message B",
        "cause_traceback": ["B"],
    }
    upsert_first_rows_response(
        "b",
        "config",
        "split",
        b_details,
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "ErrorCodeB",
        b_details,
    )
    c_details = {
        "error": "error C",
        "cause_exception": "ExceptionC",
        "cause_message": "Cause message C",
        "cause_traceback": ["C"],
    }
    upsert_first_rows_response(
        "c",
        "config",
        "split",
        {
            "error": c_details["error"],
        },
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "ErrorCodeC",
        c_details,
    )
    response = get_cache_reports_first_rows("", 2)
    assert response["cache_reports"] == [
        {"dataset": "a", "config": "config", "split": "split", "http_status": HTTPStatus.OK.value, "error_code": None},
        {
            "dataset": "b",
            "config": "config",
            "split": "split",
            "http_status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
            "error_code": "ErrorCodeB",
        },
    ]
    assert response["next_cursor"] != ""
    next_cursor = response["next_cursor"]

    response = get_cache_reports_first_rows(next_cursor, 2)
    assert response == {
        "cache_reports": [
            {
                "dataset": "c",
                "config": "config",
                "split": "split",
                "http_status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
                "error_code": "ErrorCodeC",
            },
        ],
        "next_cursor": "",
    }

    with pytest.raises(InvalidCursor):
        get_cache_reports_first_rows("not an objectid", 2)
    with pytest.raises(InvalidLimit):
        get_cache_reports_first_rows(next_cursor, -1)
    with pytest.raises(InvalidLimit):
        get_cache_reports_first_rows(next_cursor, 0)


@pytest.mark.parametrize("num_entries", [100, 1_000])
def test_stress_get_cache_reports_first_rows(num_entries: int) -> None:
    MAX_SECONDS = 0.1
    assert get_cache_reports_first_rows("", 2) == {"cache_reports": [], "next_cursor": ""}
    split_names = [f"split{i}" for i in range(num_entries)]
    for split_name in split_names:
        upsert_first_rows_response(
            "dataset",
            "config",
            split_name,
            {"key": "value"},
            HTTPStatus.OK,
        )

    next_cursor = ""
    is_first: bool = True
    while next_cursor != "" or is_first:
        start = process_time()
        is_first = False
        response = get_cache_reports_first_rows(next_cursor, 100)
        next_cursor = response["next_cursor"]
        assert process_time() - start < MAX_SECONDS


def test_get_cache_reports_features() -> None:
    assert get_cache_reports_features("", 2) == {"cache_reports": [], "next_cursor": ""}
    upsert_first_rows_response(
        "a",
        "config",
        "split",
        {"key": "value"},
        HTTPStatus.OK,
    )
    b_details = {
        "error": "error B",
        "cause_exception": "ExceptionB",
        "cause_message": "Cause message B",
        "cause_traceback": ["B"],
    }
    upsert_first_rows_response(
        "b",
        "config",
        "split",
        b_details,
        HTTPStatus.INTERNAL_SERVER_ERROR,
        "ErrorCodeB",
        b_details,
    )
    upsert_first_rows_response(
        "c",
        "config",
        "split",
        {"features": "value"},
        HTTPStatus.OK,
    )
    upsert_first_rows_response(
        "d",
        "config",
        "split",
        {"features": "value2"},
        HTTPStatus.OK,
    )
    upsert_first_rows_response(
        "e",
        "config",
        "split",
        {"features": "value3"},
        HTTPStatus.OK,
    )
    response = get_cache_reports_features("", 2)
    assert response["cache_reports"] == [
        {"dataset": "c", "config": "config", "split": "split", "features": "value"},
        {"dataset": "d", "config": "config", "split": "split", "features": "value2"},
    ]
    assert response["next_cursor"] != ""
    next_cursor = response["next_cursor"]

    response = get_cache_reports_features(next_cursor, 2)
    assert response == {
        "cache_reports": [
            {"dataset": "e", "config": "config", "split": "split", "features": "value3"},
        ],
        "next_cursor": "",
    }

    with pytest.raises(InvalidCursor):
        get_cache_reports_features("not an objectid", 2)
    with pytest.raises(InvalidLimit):
        get_cache_reports_features(next_cursor, -1)
    with pytest.raises(InvalidLimit):
        get_cache_reports_features(next_cursor, 0)


@pytest.mark.parametrize("num_entries", [100, 1_000])
def test_stress_get_cache_reports_features(num_entries: int) -> None:
    MAX_SECONDS = 0.1
    assert get_cache_reports_features("", 2) == {"cache_reports": [], "next_cursor": ""}
    split_names = [f"split{i}" for i in range(num_entries)]
    for split_name in split_names:
        upsert_first_rows_response(
            "dataset",
            "config",
            split_name,
            {"features": "value"},
            HTTPStatus.OK,
        )

    next_cursor = ""
    is_first: bool = True
    while next_cursor != "" or is_first:
        start = process_time()
        is_first = False
        response = get_cache_reports_features(next_cursor, 100)
        next_cursor = response["next_cursor"]
        assert process_time() - start < MAX_SECONDS
