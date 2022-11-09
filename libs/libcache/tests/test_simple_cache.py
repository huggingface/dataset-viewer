# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from time import process_time
from typing import Optional

import pytest
from pymongo.errors import DocumentTooLarge

from libcache.simple_cache import (
    DoesNotExist,
    InvalidCursor,
    InvalidLimit,
    _clean_database,
    delete_responses,
    get_cache_reports,
    get_response,
    get_responses_count_by_kind_status_and_error_code,
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


def test_count_by_status_and_error_code() -> None:
    assert "OK" not in get_responses_count_by_kind_status_and_error_code()

    upsert_response(
        kind="test_kind",
        dataset="test_dataset",
        content={"key": "value"},
        http_status=HTTPStatus.OK,
    )

    assert get_responses_count_by_kind_status_and_error_code() == [
        {"kind": "test_kind", "http_status": 200, "error_code": None, "count": 1}
    ]

    upsert_response(
        kind="test_kind2",
        dataset="test_dataset",
        config="test_config",
        split="test_split",
        content={
            "key": "value",
        },
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        error_code="error_code",
    )

    assert get_responses_count_by_kind_status_and_error_code() == [
        {"kind": "test_kind", "http_status": 200, "error_code": None, "count": 1},
        {"kind": "test_kind2", "http_status": 500, "error_code": "error_code", "count": 1},
    ]


def test_get_cache_reports() -> None:
    kind = "test_kind"
    kind_2 = "test_kind_2"
    assert get_cache_reports(kind=kind, cursor="", limit=2) == {"cache_reports": [], "next_cursor": ""}

    dataset_a = "test_dataset_a"
    content_a = {"key": "a"}
    http_status_a = HTTPStatus.OK
    upsert_response(
        kind=kind,
        dataset=dataset_a,
        content=content_a,
        http_status=http_status_a,
    )

    dataset_b = "test_dataset_b"
    config_b = "test_config_b"
    content_b = {"key": "b"}
    http_status_b = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code_b = "error_code_b"
    details_b = {
        "error": "error b",
    }
    worker_version_b = "0.1.2"
    dataset_git_revision_b = "123456"
    upsert_response(
        kind=kind,
        dataset=dataset_b,
        config=config_b,
        content=content_b,
        details=details_b,
        http_status=http_status_b,
        error_code=error_code_b,
        worker_version=worker_version_b,
        dataset_git_revision=dataset_git_revision_b,
    )

    dataset_c = "test_dataset_c"
    config_c = "test_config_c"
    split_c = "test_split_c"
    content_c = {"key": "c"}
    http_status_c = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code_c = "error_code_c"
    details_c = {
        "error": "error c",
    }
    upsert_response(
        kind=kind,
        dataset=dataset_c,
        config=config_c,
        split=split_c,
        content=content_c,
        details=details_c,
        http_status=http_status_c,
        error_code=error_code_c,
    )
    mark_responses_as_stale(kind=kind, dataset=dataset_c, config=config_c, split=split_c)

    upsert_response(
        kind=kind_2,
        dataset=dataset_c,
        content=content_c,
        details=details_c,
        http_status=http_status_c,
        error_code=error_code_c,
    )

    response = get_cache_reports(kind=kind, cursor="", limit=2)
    assert response["cache_reports"] == [
        {
            "kind": kind,
            "dataset": dataset_a,
            "config": None,
            "split": None,
            "http_status": http_status_a.value,
            "error_code": None,
            "worker_version": None,
            "dataset_git_revision": None,
            "stale": False,
        },
        {
            "kind": kind,
            "dataset": dataset_b,
            "config": config_b,
            "split": None,
            "http_status": http_status_b.value,
            "error_code": error_code_b,
            "worker_version": worker_version_b,
            "dataset_git_revision": dataset_git_revision_b,
            "stale": False,
        },
    ]
    assert response["next_cursor"] != ""
    next_cursor = response["next_cursor"]

    response = get_cache_reports(kind=kind, cursor=next_cursor, limit=2)
    assert response == {
        "cache_reports": [
            {
                "kind": kind,
                "dataset": dataset_c,
                "config": config_c,
                "split": split_c,
                "http_status": http_status_c.value,
                "error_code": error_code_c,
                "worker_version": None,
                "dataset_git_revision": None,
                "stale": True,
            },
        ],
        "next_cursor": "",
    }

    with pytest.raises(InvalidCursor):
        get_cache_reports(kind=kind, cursor="not an objectid", limit=2)
    with pytest.raises(InvalidLimit):
        get_cache_reports(kind=kind, cursor=next_cursor, limit=-1)
    with pytest.raises(InvalidLimit):
        get_cache_reports(kind=kind, cursor=next_cursor, limit=0)


@pytest.mark.parametrize("num_entries", [1, 10, 100, 1_000])
def test_stress_get_cache_reports(num_entries: int) -> None:
    MAX_SECONDS = 0.1
    kind = "test_kind"
    content = {"key": "value"}
    http_status = HTTPStatus.OK
    splits = [f"split{i}" for i in range(num_entries)]
    for split in splits:
        upsert_response(
            kind=kind,
            dataset="dataset",
            config="config",
            split=split,
            content=content,
            http_status=http_status,
        )

    next_cursor = ""
    is_first: bool = True
    while next_cursor != "" or is_first:
        start = process_time()
        is_first = False
        response = get_cache_reports(kind=kind, cursor=next_cursor, limit=100)
        next_cursor = response["next_cursor"]
        assert process_time() - start < MAX_SECONDS
