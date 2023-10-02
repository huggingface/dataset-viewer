# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Mapping
from datetime import datetime
from http import HTTPStatus
from time import process_time
from typing import Any, Optional, TypedDict

import pytest
from pymongo.errors import DocumentTooLarge

from libcommon.resources import CacheMongoResource
from libcommon.simple_cache import (
    CachedArtifactError,
    CachedResponseDocument,
    CacheEntryDoesNotExistError,
    CacheReportsPage,
    CacheReportsWithContentPage,
    CacheTotalMetricDocument,
    DatasetWithRevision,
    InvalidCursor,
    InvalidLimit,
    delete_dataset_responses,
    delete_response,
    fetch_names,
    get_best_response,
    get_cache_reports,
    get_cache_reports_with_content,
    get_dataset_responses_without_content_for_kind,
    get_datasets_with_last_updated_kind,
    get_outdated_split_full_names_for_step,
    get_response,
    get_response_with_details,
    get_response_without_content,
    get_responses_count_by_kind_status_and_error_code,
    has_any_successful_response,
    upsert_response,
)
from libcommon.utils import get_datetime

from .utils import CONFIG_NAME_1, CONFIG_NAME_2, CONTENT_ERROR, DATASET_NAME


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


def test_insert_null_values() -> None:
    kind = "test_kind"
    dataset_a = "test_dataset_a"
    dataset_b = "test_dataset_b"
    dataset_c = "test_dataset_c"
    config = None
    split = None
    content = {"some": "content"}
    http_status = HTTPStatus.OK

    CachedResponseDocument.objects(kind=kind, dataset=dataset_a, config=config, split=split).upsert_one(
        content=content,
        http_status=http_status,
    )
    assert CachedResponseDocument.objects.count() == 1
    cached_response = CachedResponseDocument.objects.get()
    assert cached_response is not None
    assert cached_response.config is None
    assert "config" not in cached_response.to_json()
    cached_response.validate()

    CachedResponseDocument(
        kind=kind, dataset=dataset_b, config=config, split=split, content=content, http_status=http_status
    ).save()
    assert CachedResponseDocument.objects.count() == 2
    cached_response = CachedResponseDocument.objects(dataset=dataset_b).get()
    assert cached_response is not None
    assert cached_response.config is None
    assert "config" not in cached_response.to_json()

    coll = CachedResponseDocument._get_collection()
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
    assert CachedResponseDocument.objects.count() == 3
    cached_response = CachedResponseDocument.objects(dataset=dataset_c).get()
    assert cached_response is not None
    assert cached_response.config is None
    assert "config" not in cached_response.to_json()


def assert_metric(http_status: HTTPStatus, error_code: Optional[str], kind: str, total: int) -> None:
    metric = CacheTotalMetricDocument.objects(http_status=http_status, error_code=error_code, kind=kind).first()
    assert metric is not None
    assert metric.total == total


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

    assert CacheTotalMetricDocument.objects().count() == 0
    upsert_response(kind=kind, dataset=dataset, config=config, split=split, content=content, http_status=HTTPStatus.OK)
    cached_response = get_response(kind=kind, dataset=dataset, config=config, split=split)
    assert cached_response == {
        "http_status": HTTPStatus.OK,
        "content": content,
        "error_code": None,
        "job_runner_version": None,
        "dataset_git_revision": None,
        "progress": None,
    }
    cached_response_without_content = get_response_without_content(
        kind=kind, dataset=dataset, config=config, split=split
    )
    assert cached_response_without_content == {
        "http_status": HTTPStatus.OK,
        "error_code": None,
        "job_runner_version": None,
        "dataset_git_revision": None,
        "progress": None,
    }

    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=1)

    # ensure it's idempotent
    upsert_response(kind=kind, dataset=dataset, config=config, split=split, content=content, http_status=HTTPStatus.OK)
    cached_response2 = get_response(kind=kind, dataset=dataset, config=config, split=split)
    assert cached_response2 == cached_response

    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=1)

    another_config = "another_config"
    upsert_response(
        kind=kind, dataset=dataset, config=another_config, split=split, content=content, http_status=HTTPStatus.OK
    )
    get_response(kind=kind, dataset=dataset, config=config, split=split)

    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=2)

    delete_dataset_responses(dataset=dataset)

    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=0)

    with pytest.raises(CacheEntryDoesNotExistError):
        get_response(kind=kind, dataset=dataset, config=config, split=split)

    error_code = "error_code"
    job_runner_version = 0
    dataset_git_revision = "123456"
    upsert_response(
        kind=kind,
        dataset=dataset,
        config=config,
        split=split,
        content=content,
        http_status=HTTPStatus.BAD_REQUEST,
        error_code=error_code,
        job_runner_version=job_runner_version,
        dataset_git_revision=dataset_git_revision,
    )

    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=0)
    assert_metric(http_status=HTTPStatus.BAD_REQUEST, error_code=error_code, kind=kind, total=1)

    cached_response3 = get_response(kind=kind, dataset=dataset, config=config, split=split)
    assert cached_response3 == {
        "http_status": HTTPStatus.BAD_REQUEST,
        "content": content,
        "error_code": error_code,
        "job_runner_version": job_runner_version,
        "dataset_git_revision": dataset_git_revision,
        "progress": None,
    }


def test_delete_response() -> None:
    kind = "test_kind"
    dataset_a = "test_dataset_a"
    dataset_b = "test_dataset_b"
    config = None
    split = "test_split"
    upsert_response(kind=kind, dataset=dataset_a, config=config, split=split, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=kind, dataset=dataset_b, config=config, split=split, content={}, http_status=HTTPStatus.OK)
    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=2)

    get_response(kind=kind, dataset=dataset_a, config=config, split=split)
    get_response(kind=kind, dataset=dataset_b, config=config, split=split)
    delete_response(kind=kind, dataset=dataset_a, config=config, split=split)
    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=1)
    with pytest.raises(CacheEntryDoesNotExistError):
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
    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind_a, total=2)
    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind_b, total=1)
    get_response(kind=kind_a, dataset=dataset_a)
    get_response(kind=kind_b, dataset=dataset_a, config=config, split=split)
    get_response(kind=kind_a, dataset=dataset_b)
    delete_dataset_responses(dataset=dataset_a)
    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind_a, total=1)
    assert_metric(http_status=HTTPStatus.OK, error_code=None, kind=kind_b, total=0)
    with pytest.raises(CacheEntryDoesNotExistError):
        get_response(kind=kind_a, dataset=dataset_a)
    with pytest.raises(CacheEntryDoesNotExistError):
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


def test_has_any_successful_response_empty() -> None:
    assert not has_any_successful_response(dataset="dataset", kinds=[])


def test_has_any_successful_response_two_valid_datasets() -> None:
    kind = "test_kind"
    other_kind = "other_kind"
    dataset_a = "test_dataset_a"
    dataset_b = "test_dataset_b"
    upsert_response(kind=kind, dataset=dataset_a, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=kind, dataset=dataset_b, content={}, http_status=HTTPStatus.OK)
    assert has_any_successful_response(dataset=dataset_a, kinds=[kind])
    assert has_any_successful_response(dataset=dataset_b, kinds=[kind])
    assert not has_any_successful_response(dataset=dataset_b, kinds=[other_kind])
    assert has_any_successful_response(dataset=dataset_b, kinds=[kind, other_kind])


def test_has_any_successful_response_two_valid_kinds() -> None:
    kind_a = "test_kind_a"
    kind_b = "test_kind_b"
    dataset = "test_dataset"
    upsert_response(kind=kind_a, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    upsert_response(kind=kind_b, dataset=dataset, content={}, http_status=HTTPStatus.OK)
    assert has_any_successful_response(dataset=dataset, kinds=[kind_a, kind_b])


def test_has_any_successful_response_at_least_one_valid_response() -> None:
    kind_a = "test_kind_a"
    kind_b = "test_kind_b"
    dataset = "test_dataset"
    config = "test_config"
    upsert_response(kind=kind_a, dataset=dataset, config=config, content={}, http_status=HTTPStatus.OK)
    upsert_response(
        kind=kind_b, dataset=dataset, config=config, content={}, http_status=HTTPStatus.INTERNAL_SERVER_ERROR
    )
    assert has_any_successful_response(dataset=dataset, config=config, kinds=[kind_a, kind_b])


def test_has_any_successful_response_only_invalid_responses() -> None:
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
    assert not has_any_successful_response(dataset=dataset, kinds=[kind])


def test_count_by_status_and_error_code() -> None:
    assert not get_responses_count_by_kind_status_and_error_code()

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

    metrics = get_responses_count_by_kind_status_and_error_code()
    assert len(metrics) == 2
    assert {"kind": "test_kind", "http_status": 200, "error_code": None, "count": 1} in metrics
    assert {"kind": "test_kind2", "http_status": 500, "error_code": "error_code", "count": 1} in metrics


def test_get_cache_reports() -> None:
    kind = "test_kind"
    kind_2 = "test_kind_2"
    expected_cache_reports: CacheReportsPage = {"cache_reports": [], "next_cursor": ""}
    assert get_cache_reports(kind=kind, cursor="", limit=2) == expected_cache_reports
    expected_cache_reports_with_content: CacheReportsWithContentPage = {
        "cache_reports_with_content": [],
        "next_cursor": "",
    }
    assert get_cache_reports_with_content(kind=kind, cursor="", limit=2) == expected_cache_reports_with_content

    dataset_a = "test_dataset_a"
    content_a = {"key": "a"}
    http_status_a = HTTPStatus.OK
    updated_at_a = datetime(2020, 1, 1, 0, 0, 0)
    upsert_response(
        kind=kind,
        dataset=dataset_a,
        content=content_a,
        http_status=http_status_a,
        updated_at=updated_at_a,
    )

    dataset_b = "test_dataset_b"
    config_b = "test_config_b"
    content_b = {"key": "b"}
    http_status_b = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code_b = "error_code_b"
    details_b = {
        "error": "error b",
    }
    job_runner_version_b = 0
    dataset_git_revision_b = "123456"
    updated_at_b = datetime(2020, 1, 1, 0, 0, 1)
    upsert_response(
        kind=kind,
        dataset=dataset_b,
        config=config_b,
        content=content_b,
        details=details_b,
        http_status=http_status_b,
        error_code=error_code_b,
        job_runner_version=job_runner_version_b,
        dataset_git_revision=dataset_git_revision_b,
        updated_at=updated_at_b,
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
    updated_at_c = datetime(2020, 1, 1, 0, 0, 2)
    upsert_response(
        kind=kind,
        dataset=dataset_c,
        config=config_c,
        split=split_c,
        content=content_c,
        details=details_c,
        http_status=http_status_c,
        error_code=error_code_c,
        updated_at=updated_at_c,
    )
    upsert_response(
        kind=kind_2,
        dataset=dataset_c,
        content=content_c,
        details=details_c,
        http_status=http_status_c,
        error_code=error_code_c,
        updated_at=updated_at_c,
    )
    upsert_response(
        kind=kind_2,
        dataset=dataset_c,
        config=config_c,
        split=split_c,
        content=content_c,
        details=details_c,
        http_status=http_status_c,
        error_code=error_code_c,
        updated_at=updated_at_c,
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
            "details": {},
            "updated_at": updated_at_a,
            "job_runner_version": None,
            "dataset_git_revision": None,
            "progress": None,
        },
        {
            "kind": kind,
            "dataset": dataset_b,
            "config": config_b,
            "split": None,
            "http_status": http_status_b.value,
            "error_code": error_code_b,
            "details": details_b,
            "updated_at": updated_at_b,
            "job_runner_version": job_runner_version_b,
            "dataset_git_revision": dataset_git_revision_b,
            "progress": None,
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
                "details": details_c,
                "updated_at": updated_at_c,
                "job_runner_version": None,
                "dataset_git_revision": None,
                "progress": None,
            },
        ],
        "next_cursor": "",
    }

    response_with_content = get_cache_reports_with_content(kind=kind, cursor="", limit=2)
    # redact the response to make it simpler to compare with the expected
    assert response_with_content["cache_reports_with_content"] == [
        {
            "kind": kind,
            "dataset": dataset_a,
            "config": None,
            "split": None,
            "http_status": http_status_a.value,
            "error_code": None,
            "content": content_a,
            "job_runner_version": None,
            "dataset_git_revision": None,
            "details": {},
            "updated_at": updated_at_a,
            "progress": None,
        },
        {
            "kind": kind,
            "dataset": dataset_b,
            "config": config_b,
            "split": None,
            "http_status": http_status_b.value,
            "error_code": error_code_b,
            "content": content_b,
            "job_runner_version": job_runner_version_b,
            "dataset_git_revision": dataset_git_revision_b,
            "details": details_b,
            "updated_at": updated_at_b,
            "progress": None,
        },
    ]
    assert response_with_content["next_cursor"] != ""
    next_cursor = response_with_content["next_cursor"]
    response_with_content = get_cache_reports_with_content(kind=kind, cursor=next_cursor, limit=2)
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
                "job_runner_version": None,
                "dataset_git_revision": None,
                "details": details_c,
                "updated_at": updated_at_c,
                "progress": None,
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

    result_a = get_dataset_responses_without_content_for_kind(kind=kind, dataset=dataset_a)
    assert len(result_a) == 1
    assert result_a[0]["http_status"] == HTTPStatus.OK.value
    assert result_a[0]["error_code"] is None
    assert result_a[0]["details"] == {}

    assert not get_dataset_responses_without_content_for_kind(kind=kind_2, dataset=dataset_a)

    result_c = get_dataset_responses_without_content_for_kind(kind=kind_2, dataset=dataset_c)
    assert len(result_c) == 2
    for result in result_c:
        assert result["http_status"] == http_status_c.value
        assert result["error_code"] == error_code_c
        assert result["details"] == details_c
        assert result["updated_at"] == updated_at_c


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


def test_get_outdated_split_full_names_for_step() -> None:
    kind = "kind"
    current_version = 2
    minor_version = 1

    result = get_outdated_split_full_names_for_step(kind=kind, current_version=current_version)
    upsert_response(
        kind=kind,
        dataset="dataset_with_current_version",
        content={},
        http_status=HTTPStatus.OK,
        job_runner_version=current_version,
    )
    assert not result

    upsert_response(
        kind=kind,
        dataset="dataset_with_minor_version",
        content={},
        http_status=HTTPStatus.OK,
        job_runner_version=minor_version,
    )
    result = get_outdated_split_full_names_for_step(kind=kind, current_version=current_version)
    assert result
    assert len(result) == 1


class EntrySpec(TypedDict):
    kind: str
    dataset: str
    config: Optional[str]
    http_status: HTTPStatus
    progress: Optional[float]


@pytest.mark.parametrize(
    "selected_entries,kinds,dataset,config,best_entry",
    [
        # Best means:
        # - the first success response with progress=1.0 is returned
        (["ok1"], ["kind1"], "dataset", None, "ok1"),
        (["ok_config1"], ["kind1"], "dataset", "config", "ok_config1"),
        (["ok1", "ok2"], ["kind1", "kind2"], "dataset", None, "ok1"),
        (["ok1", "ok2"], ["kind2", "kind1"], "dataset", None, "ok2"),
        (["partial1", "ok2"], ["kind1", "kind2"], "dataset", None, "ok2"),
        (["error1", "ok2"], ["kind1", "kind2"], "dataset", None, "ok2"),
        # - if no success response with progress=1.0 is found, the success response with the highest progress is
        #  returned
        (["partial1", "partial2"], ["kind1", "kind2"], "dataset", None, "partial2"),
        (["partial1", "error2"], ["kind1", "kind2"], "dataset", None, "partial1"),
        # - if no success response is found, the first error response is returned
        (["error1", "error2"], ["kind1", "kind2"], "dataset", None, "error1"),
        (["error1", "error2"], ["kind2", "kind1"], "dataset", None, "error2"),
        # - if no response is found, an error response is returned
        ([], ["kind1"], "dataset", None, "cache_miss"),
        (["ok_config1"], ["kind1"], "dataset", None, "cache_miss"),
        (["ok1"], ["kind1"], "dataset", "config", "cache_miss"),
    ],
)
def test_get_best_response(
    selected_entries: list[str], kinds: list[str], dataset: str, config: Optional[str], best_entry: str
) -> None:
    # arrange
    entries: dict[str, EntrySpec] = {
        "ok1": {
            "kind": "kind1",
            "dataset": "dataset",
            "config": None,
            "http_status": HTTPStatus.OK,
            "progress": 1.0,
        },
        "ok2": {
            "kind": "kind2",
            "dataset": "dataset",
            "config": None,
            "http_status": HTTPStatus.OK,
            "progress": 1.0,
        },
        "partial1": {
            "kind": "kind1",
            "dataset": "dataset",
            "config": None,
            "http_status": HTTPStatus.OK,
            "progress": 0,
        },
        "partial2": {
            "kind": "kind2",
            "dataset": "dataset",
            "config": None,
            "http_status": HTTPStatus.OK,
            "progress": 0.5,
        },
        "ok_config1": {
            "kind": "kind1",
            "dataset": "dataset",
            "config": "config",
            "http_status": HTTPStatus.OK,
            "progress": 1.0,
        },
        "error1": {
            "kind": "kind1",
            "dataset": "dataset",
            "config": None,
            "http_status": HTTPStatus.INTERNAL_SERVER_ERROR,
            "progress": 1.0,
        },
        "error2": {
            "kind": "kind2",
            "dataset": "dataset",
            "config": None,
            "http_status": HTTPStatus.NOT_FOUND,
            "progress": 1.0,
        },
        "cache_miss": {
            "kind": "kind1",
            "dataset": "dataset",
            "config": None,
            "http_status": HTTPStatus.NOT_FOUND,
            "progress": None,
        },
    }

    for entry in selected_entries:
        upsert_response(
            kind=entries[entry]["kind"],
            dataset=entries[entry]["dataset"],
            config=entries[entry]["config"],
            http_status=entries[entry]["http_status"],
            progress=entries[entry]["progress"],
            content={"error": "some_error"} if (entries[entry]["http_status"] >= HTTPStatus.BAD_REQUEST.value) else {},
        )

    # act
    best_response = get_best_response(kinds, dataset, config)

    # assert
    assert best_response.kind == entries[best_entry]["kind"]
    assert ("error" in best_response.response["content"]) is (
        entries[best_entry]["http_status"] >= HTTPStatus.BAD_REQUEST.value
    )
    assert best_response.response["http_status"] == entries[best_entry]["http_status"].value
    assert best_response.response["progress"] == entries[best_entry]["progress"]


def test_cached_artifact_error() -> None:
    dataset = "dataset"
    config = "config"
    split = "split"
    kind = "cache_kind"
    error_code = "ErrorCode"
    error_message = "error message"
    cause_exception = "CauseException"
    cause_message = "cause message"
    cause_traceback = ["traceback1", "traceback2"]
    details = {
        "error": error_message,
        "cause_exception": cause_exception,
        "cause_message": cause_message,
        "cause_traceback": cause_traceback,
    }
    content = {"error": error_message}
    job_runner_version = 1
    dataset_git_revision = "dataset_git_revision"
    progress = 1.0
    upsert_response(
        kind=kind,
        dataset=dataset,
        config=config,
        split=split,
        content=content,
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        error_code=error_code,
        details=details,
        job_runner_version=job_runner_version,
        dataset_git_revision=dataset_git_revision,
        progress=progress,
    )
    response = get_response_with_details(kind=kind, dataset=dataset, config=config, split=split)
    error = CachedArtifactError(
        message="Previous step error",
        kind=kind,
        dataset=dataset,
        config=config,
        split=split,
        cache_entry_with_details=response,
    )

    assert error.cache_entry_with_details["content"] == content
    assert error.cache_entry_with_details["http_status"] == HTTPStatus.INTERNAL_SERVER_ERROR
    assert error.cache_entry_with_details["error_code"] == error_code
    assert error.enhanced_details == {
        "error": error_message,
        "cause_exception": cause_exception,
        "cause_message": cause_message,
        "cause_traceback": cause_traceback,
        "copied_from_artifact": {
            "kind": kind,
            "dataset": dataset,
            "config": config,
            "split": split,
        },
    }


class ResponseSpec(TypedDict):
    content: Mapping[str, Any]
    http_status: HTTPStatus


CACHE_KIND_A = "cache_kind_a"
CACHE_KIND_B = "cache_kind_b"
NAMES = ["name_1", "name_2", "name_3"]
NAME_FIELD = "name"
NAMES_FIELD = "names"
NAMES_RESPONSE_OK = ResponseSpec(
    content={NAMES_FIELD: [{NAME_FIELD: name} for name in NAMES]}, http_status=HTTPStatus.OK
)
RESPONSE_ERROR = ResponseSpec(content=CONTENT_ERROR, http_status=HTTPStatus.INTERNAL_SERVER_ERROR)


@pytest.mark.parametrize(
    "cache_kinds,response_spec_by_kind,expected_names",
    [
        ([], {}, []),
        ([CACHE_KIND_A], {}, []),
        ([CACHE_KIND_A], {CACHE_KIND_A: RESPONSE_ERROR}, []),
        ([CACHE_KIND_A], {CACHE_KIND_A: NAMES_RESPONSE_OK}, NAMES),
        ([CACHE_KIND_A, CACHE_KIND_B], {CACHE_KIND_A: NAMES_RESPONSE_OK}, NAMES),
        ([CACHE_KIND_A, CACHE_KIND_B], {CACHE_KIND_A: NAMES_RESPONSE_OK, CACHE_KIND_B: RESPONSE_ERROR}, NAMES),
        ([CACHE_KIND_A, CACHE_KIND_B], {CACHE_KIND_A: NAMES_RESPONSE_OK, CACHE_KIND_B: NAMES_RESPONSE_OK}, NAMES),
        ([CACHE_KIND_A, CACHE_KIND_B], {CACHE_KIND_A: RESPONSE_ERROR, CACHE_KIND_B: RESPONSE_ERROR}, []),
    ],
)
def test_fetch_names(
    cache_kinds: list[str],
    response_spec_by_kind: Mapping[str, Mapping[str, Any]],
    expected_names: list[str],
) -> None:
    for kind, response_spec in response_spec_by_kind.items():
        upsert_response(
            kind=kind,
            dataset=DATASET_NAME,
            config=CONFIG_NAME_1,
            split=None,
            content=response_spec["content"],
            http_status=response_spec["http_status"],
        )
    assert (
        fetch_names(
            dataset=DATASET_NAME,
            config=CONFIG_NAME_1,
            cache_kinds=cache_kinds,
            names_field=NAMES_FIELD,
            name_field=NAME_FIELD,
        )
        == expected_names
    )


class Entry(TypedDict):
    kind: str
    dataset: str
    config: str
    http_status: HTTPStatus
    updated_at: datetime
    dataset_git_revision: Optional[str]


DAYS = 2
BEFORE = get_datetime(days=DAYS + 1)
AFTER_1 = get_datetime(days=DAYS - 1)
AFTER_2 = get_datetime(days=DAYS - 1.5)
REVISION_A = "revision_A"
REVISION_B = "revision_B"
ENTRY_1: Entry = {
    "kind": CACHE_KIND_A,
    "dataset": DATASET_NAME,
    "config": CONFIG_NAME_1,
    "http_status": HTTPStatus.OK,
    "updated_at": AFTER_1,
    "dataset_git_revision": REVISION_A,
}
ENTRY_2: Entry = {
    "kind": CACHE_KIND_B,
    "dataset": DATASET_NAME,
    "config": CONFIG_NAME_1,
    "http_status": HTTPStatus.OK,
    "updated_at": AFTER_1,
    "dataset_git_revision": None,
}
DATASET_2 = f"{DATASET_NAME}_2"
ENTRY_3: Entry = {
    "kind": CACHE_KIND_A,
    "dataset": DATASET_2,
    "config": CONFIG_NAME_1,
    "http_status": HTTPStatus.OK,
    "updated_at": AFTER_1,
    "dataset_git_revision": None,
}
ENTRY_4: Entry = {
    "kind": CACHE_KIND_A,
    "dataset": DATASET_NAME,
    "config": CONFIG_NAME_2,
    "http_status": HTTPStatus.OK,
    "updated_at": AFTER_2,
    "dataset_git_revision": REVISION_B,
}
ENTRY_5: Entry = {
    "kind": CACHE_KIND_A,
    "dataset": DATASET_NAME,
    "config": CONFIG_NAME_1,
    "http_status": HTTPStatus.INTERNAL_SERVER_ERROR,
    "updated_at": AFTER_1,
    "dataset_git_revision": None,
}
ENTRY_6: Entry = {
    "kind": CACHE_KIND_A,
    "dataset": DATASET_NAME,
    "config": CONFIG_NAME_1,
    "http_status": HTTPStatus.OK,
    "updated_at": BEFORE,
    "dataset_git_revision": None,
}

DATASET_REV_A = DatasetWithRevision(dataset=DATASET_NAME, revision=REVISION_A)
DATASET_2_REV_NONE = DatasetWithRevision(dataset=DATASET_2, revision=None)
DATASET_REV_B = DatasetWithRevision(dataset=DATASET_NAME, revision=REVISION_B)


def get_dataset(dataset_with_revision: DatasetWithRevision) -> str:
    return dataset_with_revision.dataset


def assert_lists_are_equal(a: list[DatasetWithRevision], b: list[DatasetWithRevision]) -> None:
    assert sorted(a, key=get_dataset) == sorted(b, key=get_dataset)


@pytest.mark.parametrize(
    "entries,expected_datasets",
    [
        ([], []),
        ([ENTRY_1], [DATASET_REV_A]),
        ([ENTRY_2], []),
        ([ENTRY_3], [DATASET_2_REV_NONE]),
        ([ENTRY_4], [DATASET_REV_B]),
        ([ENTRY_5], []),
        ([ENTRY_6], []),
        ([ENTRY_1, ENTRY_3], [DATASET_REV_A, DATASET_2_REV_NONE]),
        ([ENTRY_1, ENTRY_4], [DATASET_REV_B]),
        ([ENTRY_1, ENTRY_2, ENTRY_3, ENTRY_4, ENTRY_5, ENTRY_6], [DATASET_REV_B, DATASET_2_REV_NONE]),
    ],
)
def test_get_datasets_with_last_updated_kind(
    entries: list[Entry], expected_datasets: list[DatasetWithRevision]
) -> None:
    for entry in entries:
        upsert_response(
            kind=entry["kind"],
            dataset=entry["dataset"],
            config=entry["config"],
            split=None,
            content={},
            http_status=entry["http_status"],
            updated_at=entry["updated_at"],
            dataset_git_revision=entry["dataset_git_revision"],
        )
    kind = CACHE_KIND_A
    days = DAYS
    assert_lists_are_equal(get_datasets_with_last_updated_kind(kind=kind, days=days), expected_datasets)
    # ^ the order is not meaningful, so we sort to make the test deterministic
