# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

# from time import process_time
from typing import Optional

import pytest
from pymongo.errors import DocumentTooLarge

from libcache.simple_cache import (
    DoesNotExist,
    _clean_database,
    delete_responses,
    get_response,
    mark_responses_as_stale,
    upsert_response,
)


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_database()


@pytest.mark.parametrize(
    "config,split",
    [
        (None, None),
        ("test_config", None),
        ("test_config", "test_split"),
    ],
)
def test_upsert_response(config: Optional[str], split: Optional[str]) -> None:
    kind = "test_kind"
    dataset = "test_dataset"
    config = None
    split = None
    content = {"some": "content"}
    upsert_response(kind=kind, dataset=dataset, config=config, split=split, content=content, http_status=HTTPStatus.OK)
    cached_response = get_response(kind=kind, dataset=dataset, config=config, split=split)
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["content"] == content
    assert cached_response["error_code"] is None
    assert cached_response["worker_version"] is None
    assert cached_response["dataset_git_revision"] is None

    # ensure it's idempotent
    upsert_response(kind=kind, dataset=dataset, config=config, split=split, content=content, http_status=HTTPStatus.OK)
    cached_response2 = get_response(kind=kind, dataset=dataset, config=config, split=split)
    assert cached_response2 == cached_response

    mark_responses_as_stale(kind=kind, dataset=dataset)
    mark_responses_as_stale(kind=kind, dataset=dataset, config=config, split=split)
    # we don't have access to the stale field
    # we also don't have access to the updated_at field

    another_config = "another_config"
    upsert_response(
        kind=kind, dataset=dataset, config=another_config, split=split, content=content, http_status=HTTPStatus.OK
    )
    delete_responses(kind=kind, dataset=dataset, config=another_config, split=split)
    get_response(kind=kind, dataset=dataset, config=config, split=split)

    delete_responses(kind=kind, dataset=dataset, config=config, split=split)
    with pytest.raises(DoesNotExist):
        get_response(kind=kind, dataset=dataset, config=config, split=split)

    mark_responses_as_stale(kind=kind, dataset=dataset, config=config, split=split)
    with pytest.raises(DoesNotExist):
        get_response(kind=kind, dataset=dataset, config=config, split=split)

    error_code = "error_code"
    worker_version = "0.1.2"
    dataset_git_revision = "123456"
    upsert_response(
        kind=kind,
        dataset=dataset,
        config=config,
        split=split,
        content=content,
        http_status=HTTPStatus.BAD_REQUEST,
        error_code=error_code,
        worker_version=worker_version,
        dataset_git_revision=dataset_git_revision,
    )

    cached_response3 = get_response(kind=kind, dataset=dataset, config=config, split=split)
    assert cached_response3["http_status"] == HTTPStatus.BAD_REQUEST
    assert cached_response3["content"] == content
    assert cached_response3["error_code"] == error_code
    assert cached_response3["worker_version"] == worker_version
    assert cached_response3["dataset_git_revision"] == dataset_git_revision


def test_big_row() -> None:
    # https://github.com/huggingface/datasets-server/issues/197
    kind = "test_kind"
    dataset = "test_dataset"
    config = "test_config"
    split = "test_split"
    big_content = {"big": "a" * 100_000_000}
    with pytest.raises(DocumentTooLarge):
        upsert_response(
            kind=kind, dataset=dataset, config=config, split=split, content=big_content, http_status=HTTPStatus.OK
        )


# def test_valid() -> None:
#     assert get_valid_datasets() == []
#     assert get_datasets_with_some_error() == []

#     upsert_splits_response(
#         "test_dataset",
#         {"key": "value"},
#         HTTPStatus.OK,
#     )

#     assert get_valid_datasets() == []
#     assert get_datasets_with_some_error() == []
#     assert is_dataset_valid("test_dataset") is False
#     assert is_dataset_valid("test_dataset2") is False
#     assert is_dataset_valid("test_dataset3") is False

#     upsert_first_rows_response(
#         "test_dataset",
#         "test_config",
#         "test_split",
#         {
#             "key": "value",
#         },
#         HTTPStatus.OK,
#     )

#     assert get_valid_datasets() == ["test_dataset"]
#     assert get_datasets_with_some_error() == []
#     assert is_dataset_valid("test_dataset") is True
#     assert is_dataset_valid("test_dataset2") is False
#     assert is_dataset_valid("test_dataset3") is False

#     upsert_splits_response(
#         "test_dataset2",
#         {"key": "value"},
#         HTTPStatus.OK,
#     )

#     assert get_valid_datasets() == ["test_dataset"]
#     assert get_datasets_with_some_error() == []
#     assert is_dataset_valid("test_dataset") is True
#     assert is_dataset_valid("test_dataset2") is False
#     assert is_dataset_valid("test_dataset3") is False

#     upsert_first_rows_response(
#         "test_dataset2",
#         "test_config2",
#         "test_split2",
#         {
#             "key": "value",
#         },
#         HTTPStatus.BAD_REQUEST,
#     )

#     assert get_valid_datasets() == ["test_dataset"]
#     assert get_datasets_with_some_error() == ["test_dataset2"]
#     assert is_dataset_valid("test_dataset") is True
#     assert is_dataset_valid("test_dataset2") is False
#     assert is_dataset_valid("test_dataset3") is False

#     upsert_first_rows_response(
#         "test_dataset2",
#         "test_config2",
#         "test_split3",
#         {
#             "key": "value",
#         },
#         HTTPStatus.OK,
#     )

#     assert get_valid_datasets() == ["test_dataset", "test_dataset2"]
#     assert get_datasets_with_some_error() == ["test_dataset2"]
#     assert is_dataset_valid("test_dataset") is True
#     assert is_dataset_valid("test_dataset2") is True
#     assert is_dataset_valid("test_dataset3") is False

#     upsert_splits_response(
#         "test_dataset3",
#         {"key": "value"},
#         HTTPStatus.BAD_REQUEST,
#     )

#     assert get_valid_datasets() == ["test_dataset", "test_dataset2"]
#     assert get_datasets_with_some_error() == ["test_dataset2", "test_dataset3"]
#     assert is_dataset_valid("test_dataset") is True
#     assert is_dataset_valid("test_dataset2") is True
#     assert is_dataset_valid("test_dataset3") is False


# def test_count_by_status_and_error_code() -> None:
#     assert "OK" not in get_splits_responses_count_by_status_and_error_code()

#     upsert_splits_response(
#         "test_dataset",
#         {"key": "value"},
#         HTTPStatus.OK,
#     )

#     assert get_splits_responses_count_by_status_and_error_code() == {"200": {None: 1}}
#     assert get_first_rows_responses_count_by_status_and_error_code() == {}

#     upsert_first_rows_response(
#         "test_dataset",
#         "test_config",
#         "test_split",
#         {
#             "key": "value",
#         },
#         HTTPStatus.OK,
#     )

#     assert get_first_rows_responses_count_by_status_and_error_code() == {"200": {None: 1}}

#     upsert_first_rows_response(
#         "test_dataset",
#         "test_config",
#         "test_split2",
#         {
#             "key": "value",
#         },
#         HTTPStatus.INTERNAL_SERVER_ERROR,
#         error_code="error_code",
#     )

#     assert get_first_rows_responses_count_by_status_and_error_code() == {
#         "200": {None: 1},
#         "500": {"error_code": 1},
#     }


# def test_get_cache_reports_splits() -> None:
#     assert get_cache_reports_splits("", 2) == {"cache_reports": [], "next_cursor": ""}
#     upsert_splits_response(
#         "a",
#         {"key": "value"},
#         HTTPStatus.OK,
#     )
#     b_details = {
#         "error": "error B",
#         "cause_exception": "ExceptionB",
#         "cause_message": "Cause message B",
#         "cause_traceback": ["B"],
#     }
#     worker_version = "0.1.2"
#     dataset_git_revision = "123456"
#     upsert_splits_response(
#         "b",
#         b_details,
#         HTTPStatus.INTERNAL_SERVER_ERROR,
#         error_code="ErrorCodeB",
#         details=b_details,
#         worker_version=worker_version,
#         dataset_git_revision=dataset_git_revision,
#     )
#     c_details = {
#         "error": "error C",
#         "cause_exception": "ExceptionC",
#         "cause_message": "Cause message C",
#         "cause_traceback": ["C"],
#     }
#     upsert_splits_response(
#         "c",
#         {
#             "error": c_details["error"],
#         },
#         HTTPStatus.INTERNAL_SERVER_ERROR,
#         "ErrorCodeC",
#         c_details,
#     )
#     response = get_cache_reports_splits("", 2)
#     assert response["cache_reports"] == [
#         {
#             "dataset": "a",
#             "http_status": HTTPStatus.OK.value,
#             "error_code": None,
#             "worker_version": None,
#             "dataset_git_revision": None,
#         },
#         {
#             "dataset": "b",
#             "http_status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
#             "error_code": "ErrorCodeB",
#             "worker_version": "0.1.2",
#             "dataset_git_revision": "123456",
#         },
#     ]
#     assert response["next_cursor"] != ""
#     next_cursor = response["next_cursor"]

#     response = get_cache_reports_splits(next_cursor, 2)
#     assert response == {
#         "cache_reports": [
#             {
#                 "dataset": "c",
#                 "http_status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
#                 "error_code": "ErrorCodeC",
#                 "worker_version": None,
#                 "dataset_git_revision": None,
#             },
#         ],
#         "next_cursor": "",
#     }

#     with pytest.raises(InvalidCursor):
#         get_cache_reports_splits("not an objectid", 2)
#     with pytest.raises(InvalidLimit):
#         get_cache_reports_splits(next_cursor, -1)
#     with pytest.raises(InvalidLimit):
#         get_cache_reports_splits(next_cursor, 0)


# def test_get_cache_reports_first_rows() -> None:
#     assert get_cache_reports_first_rows("", 2) == {"cache_reports": [], "next_cursor": ""}
#     upsert_first_rows_response(
#         "a",
#         "config",
#         "split",
#         {"key": "value"},
#         HTTPStatus.OK,
#     )
#     b_details = {
#         "error": "error B",
#         "cause_exception": "ExceptionB",
#         "cause_message": "Cause message B",
#         "cause_traceback": ["B"],
#     }
#     worker_version = "0.1.2"
#     dataset_git_revision = "123456"
#     upsert_first_rows_response(
#         "b",
#         "config",
#         "split",
#         b_details,
#         HTTPStatus.INTERNAL_SERVER_ERROR,
#         error_code="ErrorCodeB",
#         details=b_details,
#         worker_version=worker_version,
#         dataset_git_revision=dataset_git_revision,
#     )
#     c_details = {
#         "error": "error C",
#         "cause_exception": "ExceptionC",
#         "cause_message": "Cause message C",
#         "cause_traceback": ["C"],
#     }
#     upsert_first_rows_response(
#         "c",
#         "config",
#         "split",
#         {
#             "error": c_details["error"],
#         },
#         HTTPStatus.INTERNAL_SERVER_ERROR,
#         "ErrorCodeC",
#         c_details,
#     )
#     response = get_cache_reports_first_rows("", 2)
#     assert response["cache_reports"] == [
#         {
#             "dataset": "a",
#             "config": "config",
#             "split": "split",
#             "http_status": HTTPStatus.OK.value,
#             "error_code": None,
#             "worker_version": None,
#             "dataset_git_revision": None,
#         },
#         {
#             "dataset": "b",
#             "config": "config",
#             "split": "split",
#             "http_status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
#             "error_code": "ErrorCodeB",
#             "worker_version": "0.1.2",
#             "dataset_git_revision": "123456",
#         },
#     ]
#     assert response["next_cursor"] != ""
#     next_cursor = response["next_cursor"]

#     response = get_cache_reports_first_rows(next_cursor, 2)
#     assert response == {
#         "cache_reports": [
#             {
#                 "dataset": "c",
#                 "config": "config",
#                 "split": "split",
#                 "http_status": HTTPStatus.INTERNAL_SERVER_ERROR.value,
#                 "error_code": "ErrorCodeC",
#                 "worker_version": None,
#                 "dataset_git_revision": None,
#             },
#         ],
#         "next_cursor": "",
#     }

#     with pytest.raises(InvalidCursor):
#         get_cache_reports_first_rows("not an objectid", 2)
#     with pytest.raises(InvalidLimit):
#         get_cache_reports_first_rows(next_cursor, -1)
#     with pytest.raises(InvalidLimit):
#         get_cache_reports_first_rows(next_cursor, 0)


# @pytest.mark.parametrize("num_entries", [100, 1_000])
# def test_stress_get_cache_reports_first_rows(num_entries: int) -> None:
#     MAX_SECONDS = 0.1
#     assert get_cache_reports_first_rows("", 2) == {"cache_reports": [], "next_cursor": ""}
#     splits = [f"split{i}" for i in range(num_entries)]
#     for split in splits:
#         upsert_first_rows_response(
#             "dataset",
#             "config",
#             split,
#             {"key": "value"},
#             HTTPStatus.OK,
#         )

#     next_cursor = ""
#     is_first: bool = True
#     while next_cursor != "" or is_first:
#         start = process_time()
#         is_first = False
#         response = get_cache_reports_first_rows(next_cursor, 100)
#         next_cursor = response["next_cursor"]
#         assert process_time() - start < MAX_SECONDS


# def test_get_cache_reports_features() -> None:
#     assert get_cache_reports_features("", 2) == {"cache_reports": [], "next_cursor": ""}
#     upsert_first_rows_response(
#         "a",
#         "config",
#         "split",
#         {"key": "value"},
#         HTTPStatus.OK,
#     )
#     b_details = {
#         "error": "error B",
#         "cause_exception": "ExceptionB",
#         "cause_message": "Cause message B",
#         "cause_traceback": ["B"],
#     }
#     upsert_first_rows_response(
#         "b",
#         "config",
#         "split",
#         b_details,
#         HTTPStatus.INTERNAL_SERVER_ERROR,
#         "ErrorCodeB",
#         b_details,
#     )
#     upsert_first_rows_response(
#         "c",
#         "config",
#         "split",
#         {"features": "value"},
#         HTTPStatus.OK,
#     )
#     upsert_first_rows_response(
#         "d",
#         "config",
#         "split",
#         {"features": "value2"},
#         HTTPStatus.OK,
#     )
#     upsert_first_rows_response(
#         "e",
#         "config",
#         "split",
#         {"features": "value3"},
#         HTTPStatus.OK,
#     )
#     response = get_cache_reports_features("", 2)
#     assert response["cache_reports"] == [
#         {"dataset": "c", "config": "config", "split": "split", "features": "value"},
#         {"dataset": "d", "config": "config", "split": "split", "features": "value2"},
#     ]
#     assert response["next_cursor"] != ""
#     next_cursor = response["next_cursor"]

#     response = get_cache_reports_features(next_cursor, 2)
#     assert response == {
#         "cache_reports": [
#             {"dataset": "e", "config": "config", "split": "split", "features": "value3"},
#         ],
#         "next_cursor": "",
#     }

#     with pytest.raises(InvalidCursor):
#         get_cache_reports_features("not an objectid", 2)
#     with pytest.raises(InvalidLimit):
#         get_cache_reports_features(next_cursor, -1)
#     with pytest.raises(InvalidLimit):
#         get_cache_reports_features(next_cursor, 0)


# @pytest.mark.parametrize("num_entries", [100, 1_000])
# def test_stress_get_cache_reports_features(num_entries: int) -> None:
#     MAX_SECONDS = 0.1
#     assert get_cache_reports_features("", 2) == {"cache_reports": [], "next_cursor": ""}
#     splits = [f"split{i}" for i in range(num_entries)]
#     for split in splits:
#         upsert_first_rows_response(
#             "dataset",
#             "config",
#             split,
#             {"features": "value"},
#             HTTPStatus.OK,
#         )

#     next_cursor = ""
#     is_first: bool = True
#     while next_cursor != "" or is_first:
#         start = process_time()
#         is_first = False
#         response = get_cache_reports_features(next_cursor, 100)
#         next_cursor = response["next_cursor"]
#         assert process_time() - start < MAX_SECONDS
