# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from datetime import datetime
from http import HTTPStatus
from time import process_time
from typing import Optional

import pytest
from pymongo.errors import DocumentTooLarge

from libcommon.config import CacheConfig
from libcommon.simple_cache import (
    CachedResponse,
    DoesNotExist,
    InvalidCursor,
    InvalidLimit,
    SplitFullName,
    _clean_cache_database,
    delete_dataset_responses,
    delete_response,
    get_cache_reports,
    get_cache_reports_with_content,
    get_response,
    get_response_without_content,
    get_responses_count_by_kind_status_and_error_code,
    get_split_full_names_for_dataset_and_kind,
    get_valid_datasets,
    get_validity_by_kind,
    upsert_response,
)


@pytest.fixture(autouse=True)
def clean_mongo_database(cache_config: CacheConfig) -> None:
    _clean_cache_database()


def test_insert_null_values() -> None:
    kind = "test_kind"
    dataset_a = "test_dataset_a"
    dataset_b = "test_dataset_b"
    dataset_c = "test_dataset_c"
    config = None
    split = None
    content = {"some": "content"}
    http_status = HTTPStatus.OK

    CachedResponse.objects(kind=kind, dataset=dataset_a, config=config, split=split).upsert_one(
        content=content,
        http_status=http_status,
    )
    assert CachedResponse.objects.count() == 1
    cached_response = CachedResponse.objects.get()
    assert cached_response is not None
    assert cached_response.config is None
    assert "config" not in cached_response.to_json()
    cached_response.validate()

    CachedResponse(
        kind=kind, dataset=dataset_b, config=config, split=split, content=content, http_status=http_status
    ).save()
    assert CachedResponse.objects.count() == 2
    cached_response = CachedResponse.objects(dataset=dataset_b).get()
    assert cached_response is not None
    assert cached_response.config is None
    assert "config" not in cached_response.to_json()

    coll = CachedResponse._get_collection()
    coll.insert_one(
        {
            "kind": kind,
            "dataset": dataset_c,
            "config": None,
            "split": None,
            "content": content,
            "http_status": http_status,
        }
    )
    assert CachedResponse.objects.count() == 3
    cached_response = CachedResponse.objects(dataset=dataset_c).get()
    assert cached_response is not None
    assert cached_response.config is None
    assert "config" not in cached_response.to_json()


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
    assert cached_response == {
        "http_status": HTTPStatus.OK,
        "content": content,
        "error_code": None,
        "worker_version": None,
        "dataset_git_revision": None,
    }
    cached_response_without_content = get_response_without_content(
        kind=kind, dataset=dataset, config=config, split=split
    )
    assert cached_response_without_content == {
        "http_status": HTTPStatus.OK,
        "error_code": None,
        "worker_version": None,
        "dataset_git_revision": None,
    }

    # ensure it's idempotent
    upsert_response(kind=kind, dataset=dataset, config=config, split=split, content=content, http_status=HTTPStatus.OK)
    cached_response2 = get_response(kind=kind, dataset=dataset, config=config, split=split)
    assert cached_response2 == cached_response

    another_config = "another_config"
    upsert_response(
        kind=kind, dataset=dataset, config=another_config, split=split, content=content, http_status=HTTPStatus.OK
    )
    get_response(kind=kind, dataset=dataset, config=config, split=split)

    delete_dataset_responses(dataset=dataset)
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
    assert cached_response3 == {
        "http_status": HTTPStatus.BAD_REQUEST,
        "content": content,
        "error_code": error_code,
        "worker_version": worker_version,
        "dataset_git_revision": dataset_git_revision,
    }


def test_delete_response() -> None:
    kind = "test_kind"
    dataset_a = "test_dataset_a"
    dataset_b = "test_dataset_b"
    config = None
    split = "test_split"
    upsert_response(kind=kind, dataset=dataset_a, config=config, split=split, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=kind, dataset=dataset_b, config=config, split=split, content={}, http_status=HTTPStatus.OK)
    get_response(kind=kind, dataset=dataset_a, config=config, split=split)
    get_response(kind=kind, dataset=dataset_b, config=config, split=split)
    delete_response(kind=kind, dataset=dataset_a, config=config, split=split)
    with pytest.raises(DoesNotExist):
        get_response(kind=kind, dataset=dataset_a, config=config, split=split)
    get_response(kind=kind, dataset=dataset_b, config=config, split=split)


def test_delete_dataset_responses() -> None:
    kind_a = "test_kind_a"
    kind_b = "test_kind_b"
    dataset_a = "test_dataset_a"
    dataset_b = "test_dataset_b"
    config = "test_config"
    split = "test_split"
    upsert_response(kind=kind_a, dataset=dataset_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=kind_b, dataset=dataset_a, config=config, split=split, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=kind_a, dataset=dataset_b, content={}, http_status=HTTPStatus.OK)
    get_response(kind=kind_a, dataset=dataset_a)
    get_response(kind=kind_b, dataset=dataset_a, config=config, split=split)
    get_response(kind=kind_a, dataset=dataset_b)
    delete_dataset_responses(dataset=dataset_a)
    with pytest.raises(DoesNotExist):
        get_response(kind=kind_a, dataset=dataset_a)
    with pytest.raises(DoesNotExist):
        get_response(kind=kind_b, dataset=dataset_a, config=config, split=split)
    get_response(kind=kind_a, dataset=dataset_b)


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


def test_get_split_full_names_for_dataset_and_kind() -> None:
    kind_a = "test_kind_a"
    kind_b = "test_kind_b"
    dataset_a = "test_dataset_a"
    dataset_b = "test_dataset_b"
    dataset_c = "test_dataset_c"
    config_a = "test_config_a"
    config_b = "test_config_b"
    split_a = "test_split_a"
    split_b = None
    upsert_response(kind=kind_a, dataset=dataset_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(
        kind=kind_b, dataset=dataset_a, config=config_a, split=split_a, content={}, http_status=HTTPStatus.OK
    )
    upsert_response(
        kind=kind_b, dataset=dataset_a, config=config_b, split=split_a, content={}, http_status=HTTPStatus.OK
    )
    upsert_response(
        kind=kind_b, dataset=dataset_a, config=config_b, split=split_b, content={}, http_status=HTTPStatus.OK
    )
    upsert_response(kind=kind_a, dataset=dataset_b, content={}, http_status=HTTPStatus.OK)
    result = get_split_full_names_for_dataset_and_kind(dataset=dataset_a, kind=kind_a)
    expected = {SplitFullName(dataset_a, None, None)}
    assert len(result) == len(expected) and all(x in expected for x in result)
    # ^ compare the contents of the lists without caring about the order
    result = get_split_full_names_for_dataset_and_kind(dataset=dataset_a, kind=kind_b)
    expected = {
        SplitFullName(dataset_a, config_a, split_a),
        SplitFullName(dataset_a, config_b, split_b),
        SplitFullName(dataset_a, config_b, split_a),
    }
    assert len(result) == len(expected) and all(x in expected for x in result)
    # ^ compare the contents of the lists without caring about the order
    assert get_split_full_names_for_dataset_and_kind(dataset=dataset_b, kind=kind_a) == {
        SplitFullName(dataset_b, None, None)
    }
    assert get_split_full_names_for_dataset_and_kind(dataset=dataset_c, kind=kind_a) == set()


def test_get_valid_dataset_names_empty() -> None:
    assert not get_valid_datasets(kind="test_kind")


def test_get_valid_dataset_names_two_valid_datasets() -> None:
    kind = "test_kind"
    dataset_a = "test_dataset_a"
    dataset_b = "test_dataset_b"
    upsert_response(kind=kind, dataset=dataset_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=kind, dataset=dataset_b, content={}, http_status=HTTPStatus.OK)
    assert get_valid_datasets(kind=kind) == {dataset_a, dataset_b}


def test_get_valid_dataset_names_filtered_by_kind() -> None:
    kind_a = "test_kind_a"
    kind_b = "test_kind_b"
    dataset_a = "test_dataset_a"
    dataset_b = "test_dataset_b"
    upsert_response(kind=kind_a, dataset=dataset_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=kind_b, dataset=dataset_b, content={}, http_status=HTTPStatus.OK)
    assert get_valid_datasets(kind=kind_a) == {dataset_a}
    assert get_valid_datasets(kind=kind_b) == {dataset_b}


def test_get_valid_dataset_names_at_least_one_valid_response() -> None:
    kind = "test_kind"
    dataset = "test_dataset"
    config_a = "test_config_a"
    config_b = "test_config_b"
    upsert_response(kind=kind, dataset=dataset, config=config_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(
        kind=kind, dataset=dataset, config=config_b, content={}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR
    )
    assert get_valid_datasets(kind=kind) == {dataset}


def test_get_valid_dataset_names_only_invalid_responses() -> None:
    kind = "test_kind"
    dataset = "test_dataset"
    config_a = "test_config_a"
    config_b = "test_config_b"
    upsert_response(
        kind=kind, dataset=dataset, config=config_a, content={}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR
    )
    upsert_response(
        kind=kind, dataset=dataset, config=config_b, content={}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR
    )
    assert not get_valid_datasets(kind=kind)


def test_get_validity_by_kind_empty() -> None:
    assert not get_validity_by_kind(dataset="dataset")


def test_get_validity_by_kind_two_valid_datasets() -> None:
    kind = "test_kind"
    dataset_a = "test_dataset_a"
    dataset_b = "test_dataset_b"
    upsert_response(kind=kind, dataset=dataset_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=kind, dataset=dataset_b, content={}, http_status=HTTPStatus.OK)
    assert get_validity_by_kind(dataset=dataset_a) == {kind: True}
    assert get_validity_by_kind(dataset=dataset_b) == {kind: True}


def test_get_validity_by_kind_two_valid_kinds() -> None:
    kind_a = "test_kind_a"
    kind_b = "test_kind_b"
    dataset = "test_dataset"
    upsert_response(kind=kind_a, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=kind_b, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    assert get_validity_by_kind(dataset=dataset) == {kind_a: True, kind_b: True}


def test_get_validity_by_kind_at_least_one_valid_response() -> None:
    kind = "test_kind"
    dataset = "test_dataset"
    config_a = "test_config_a"
    config_b = "test_config_b"
    upsert_response(kind=kind, dataset=dataset, config=config_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(
        kind=kind, dataset=dataset, config=config_b, content={}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR
    )
    assert get_validity_by_kind(dataset=dataset) == {kind: True}


def test_get_validity_by_kind_only_invalid_responses() -> None:
    kind = "test_kind"
    dataset = "test_dataset"
    config_a = "test_config_a"
    config_b = "test_config_b"
    upsert_response(
        kind=kind, dataset=dataset, config=config_a, content={}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR
    )
    upsert_response(
        kind=kind, dataset=dataset, config=config_b, content={}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR
    )
    assert get_validity_by_kind(dataset=dataset) == {kind: False}


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
    assert get_cache_reports_with_content(kind=kind, cursor="", limit=2) == {
        "cache_reports_with_content": [],
        "next_cursor": "",
    }

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
            },
        ],
        "next_cursor": "",
    }

    response_with_content = get_cache_reports_with_content(kind=kind, cursor="", limit=2)
    # redact the response to make it simpler to compare with the expected
    REDACTED_DATE = datetime(2020, 1, 1, 0, 0, 0)
    for c in response_with_content["cache_reports_with_content"]:
        c["updated_at"] = REDACTED_DATE
    assert response_with_content["cache_reports_with_content"] == [
        {
            "kind": kind,
            "dataset": dataset_a,
            "config": None,
            "split": None,
            "http_status": http_status_a.value,
            "error_code": None,
            "content": content_a,
            "worker_version": None,
            "dataset_git_revision": None,
            "details": {},
            "updated_at": REDACTED_DATE,
        },
        {
            "kind": kind,
            "dataset": dataset_b,
            "config": config_b,
            "split": None,
            "http_status": http_status_b.value,
            "error_code": error_code_b,
            "content": content_b,
            "worker_version": worker_version_b,
            "dataset_git_revision": dataset_git_revision_b,
            "details": details_b,
            "updated_at": REDACTED_DATE,
        },
    ]
    assert response_with_content["next_cursor"] != ""
    next_cursor = response_with_content["next_cursor"]
    response_with_content = get_cache_reports_with_content(kind=kind, cursor=next_cursor, limit=2)
    for c in response_with_content["cache_reports_with_content"]:
        c["updated_at"] = REDACTED_DATE
    assert response_with_content == {
        "cache_reports_with_content": [
            {
                "kind": kind,
                "dataset": dataset_c,
                "config": config_c,
                "split": split_c,
                "http_status": http_status_c.value,
                "error_code": error_code_c,
                "content": content_c,
                "worker_version": None,
                "dataset_git_revision": None,
                "details": details_c,
                "updated_at": REDACTED_DATE,
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
