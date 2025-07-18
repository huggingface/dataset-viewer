# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Mapping
from datetime import datetime
from decimal import Decimal
from http import HTTPStatus
from time import process_time
from typing import Any, Optional, TypedDict

import pytest
from pymongo.errors import DocumentTooLarge

from libcommon.resources import CacheMongoResource
from libcommon.simple_cache import (
    CachedArtifactError,
    CachedArtifactNotFoundError,
    CachedResponseDocument,
    CacheReportsPage,
    CacheReportsWithContentPage,
    CacheTotalMetricDocument,
    InvalidCursor,
    InvalidLimit,
    delete_dataset_responses,
    delete_response,
    fetch_names,
    get_cache_reports,
    get_cache_reports_with_content,
    get_dataset_responses_without_content_for_kind,
    get_datasets_with_last_updated_kind,
    get_outdated_split_full_names_for_step,
    get_previous_step_or_raise,
    get_response,
    get_response_with_details,
    get_response_without_content,
    get_responses_count_by_kind_status_and_error_code,
    is_successful_response,
    upsert_response,
)
from libcommon.utils import get_datetime

from .utils import (
    CACHE_KIND,
    CONFIG_NAME_1,
    CONFIG_NAME_2,
    CONTENT_ERROR,
    DATASET_GIT_REVISION_A,
    DATASET_GIT_REVISION_B,
    DATASET_GIT_REVISION_C,
    DATASET_NAME,
    DATASET_NAME_A,
    DATASET_NAME_B,
    DATASET_NAME_C,
    REVISION_NAME,
)


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(
    cache_mongo_resource: CacheMongoResource,
) -> CacheMongoResource:
    return cache_mongo_resource


def test_insert_null_values() -> None:
    kind = CACHE_KIND
    dataset_a = DATASET_NAME_A
    dataset_git_revision_a = DATASET_GIT_REVISION_A
    dataset_b = DATASET_NAME_B
    dataset_c = DATASET_NAME_C
    dataset_git_revision_c = DATASET_GIT_REVISION_C
    config = None
    split = None
    content = {"some": "content"}
    http_status = HTTPStatus.OK

    CachedResponseDocument.objects(
        kind=kind,
        dataset=dataset_a,
        dataset_git_revision=dataset_git_revision_a,
        config=config,
        split=split,
    ).upsert_one(
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
        kind=kind,
        dataset=dataset_b,
        dataset_git_revision=dataset_git_revision_a,
        config=config,
        split=split,
        content=content,
        http_status=http_status,
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
            "dataset_git_revision": dataset_git_revision_c,
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


def assert_metric_entries_per_kind(http_status: HTTPStatus, error_code: Optional[str], kind: str, total: int) -> None:
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
    kind = CACHE_KIND
    dataset = DATASET_NAME
    dataset_git_revision = REVISION_NAME
    config = None
    split = None
    content = {"some": "content"}

    assert CacheTotalMetricDocument.objects().count() == 0
    upsert_response(
        kind=kind,
        dataset=dataset,
        dataset_git_revision=dataset_git_revision,
        config=config,
        split=split,
        content=content,
        http_status=HTTPStatus.OK,
    )
    cached_response = get_response(kind=kind, dataset=dataset, config=config, split=split)
    assert cached_response == {
        "http_status": HTTPStatus.OK,
        "content": content,
        "error_code": None,
        "job_runner_version": None,
        "dataset_git_revision": dataset_git_revision,
        "progress": None,
    }
    cached_response_without_content = get_response_without_content(
        kind=kind, dataset=dataset, config=config, split=split
    )
    assert cached_response_without_content == {
        "http_status": HTTPStatus.OK,
        "error_code": None,
        "job_runner_version": None,
        "dataset_git_revision": dataset_git_revision,
        "progress": None,
    }

    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=1)

    # ensure it's idempotent
    upsert_response(
        kind=kind,
        dataset=dataset,
        dataset_git_revision=dataset_git_revision,
        config=config,
        split=split,
        content=content,
        http_status=HTTPStatus.OK,
    )
    cached_response2 = get_response(kind=kind, dataset=dataset, config=config, split=split)
    assert cached_response2 == cached_response

    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=1)

    another_config = "another_config"
    upsert_response(
        kind=kind,
        dataset=dataset,
        dataset_git_revision=dataset_git_revision,
        config=another_config,
        split=split,
        content=content,
        http_status=HTTPStatus.OK,
    )
    get_response(kind=kind, dataset=dataset, config=config, split=split)

    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=2)

    delete_dataset_responses(dataset=dataset)

    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=0)

    with pytest.raises(CachedArtifactNotFoundError):
        get_response(kind=kind, dataset=dataset, config=config, split=split)

    error_code = "error_code"
    job_runner_version = 0
    upsert_response(
        kind=kind,
        dataset=dataset,
        dataset_git_revision=dataset_git_revision,
        config=config,
        split=split,
        content=content,
        http_status=HTTPStatus.BAD_REQUEST,
        error_code=error_code,
        job_runner_version=job_runner_version,
    )

    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=0)
    assert_metric_entries_per_kind(http_status=HTTPStatus.BAD_REQUEST, error_code=error_code, kind=kind, total=1)

    cached_response3 = get_response(kind=kind, dataset=dataset, config=config, split=split)
    assert cached_response3 == {
        "http_status": HTTPStatus.BAD_REQUEST,
        "content": content,
        "error_code": error_code,
        "job_runner_version": job_runner_version,
        "dataset_git_revision": dataset_git_revision,
        "progress": None,
    }


def test_upsert_response_types() -> None:
    kind = CACHE_KIND
    dataset = DATASET_NAME
    dataset_git_revision = REVISION_NAME

    now = datetime.now()
    decimal = Decimal(now.time().microsecond * 1e-6)
    content = {
        "datetime": now,  # microsecond is truncated to millisecond
        "time": now.time(),  # time it turned into a string
        "date": now.date(),  # date is turned into a string
        "decimal": decimal,  # decimal is turned into a string
    }
    upsert_response(
        kind=kind,
        dataset=dataset,
        dataset_git_revision=dataset_git_revision,
        content=content,
        http_status=HTTPStatus.OK,
    )
    cached_response = get_response(kind=kind, dataset=dataset)
    assert cached_response["content"]["datetime"] == datetime(
        now.year,
        now.month,
        now.day,
        now.hour,
        now.minute,
        now.second,
        now.microsecond // 1000 * 1000,
    )
    assert cached_response["content"]["time"] == str(now.time())
    assert cached_response["content"]["date"] == str(now.date())
    assert cached_response["content"]["decimal"] == str(decimal)


def test_delete_response() -> None:
    kind = CACHE_KIND
    dataset_a = DATASET_NAME_A
    dataset_git_revision_a = DATASET_GIT_REVISION_A
    dataset_b = DATASET_NAME_B
    dataset_git_revision_b = DATASET_GIT_REVISION_B
    config = None
    split = "test_split"
    upsert_response(
        kind=kind,
        dataset=dataset_a,
        dataset_git_revision=dataset_git_revision_a,
        config=config,
        split=split,
        content={},
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=kind,
        dataset=dataset_b,
        dataset_git_revision=dataset_git_revision_b,
        config=config,
        split=split,
        content={},
        http_status=HTTPStatus.OK,
    )
    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=2)

    get_response(kind=kind, dataset=dataset_a, config=config, split=split)
    get_response(kind=kind, dataset=dataset_b, config=config, split=split)
    delete_response(kind=kind, dataset=dataset_a, config=config, split=split)
    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind, total=1)
    with pytest.raises(CachedArtifactNotFoundError):
        get_response(kind=kind, dataset=dataset_a, config=config, split=split)
    get_response(kind=kind, dataset=dataset_b, config=config, split=split)


def test_delete_dataset_responses() -> None:
    kind_a = "test_kind_a"
    kind_b = "test_kind_b"
    dataset_a = DATASET_NAME_A
    dataset_b = DATASET_NAME_B
    dataset_git_revision_a = DATASET_GIT_REVISION_A
    dataset_git_revision_b = DATASET_GIT_REVISION_B
    config = "test_config"
    split = "test_split"
    upsert_response(
        kind=kind_a,
        dataset=dataset_a,
        dataset_git_revision=dataset_git_revision_a,
        content={},
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=kind_b,
        dataset=dataset_a,
        dataset_git_revision=dataset_git_revision_b,
        config=config,
        split=split,
        content={},
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=kind_a,
        dataset=dataset_b,
        dataset_git_revision=dataset_git_revision_b,
        content={},
        http_status=HTTPStatus.OK,
    )
    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind_a, total=2)
    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind_b, total=1)
    get_response(kind=kind_a, dataset=dataset_a)
    get_response(kind=kind_b, dataset=dataset_a, config=config, split=split)
    get_response(kind=kind_a, dataset=dataset_b)
    delete_dataset_responses(dataset=dataset_a)
    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind_a, total=1)
    assert_metric_entries_per_kind(http_status=HTTPStatus.OK, error_code=None, kind=kind_b, total=0)
    with pytest.raises(CachedArtifactNotFoundError):
        get_response(kind=kind_a, dataset=dataset_a)
    with pytest.raises(CachedArtifactNotFoundError):
        get_response(kind=kind_b, dataset=dataset_a, config=config, split=split)
    get_response(kind=kind_a, dataset=dataset_b)


def test_big_row() -> None:
    # https://github.com/huggingface/dataset-viewer/issues/197
    kind = CACHE_KIND
    dataset = DATASET_NAME
    dataset_git_revision = REVISION_NAME
    config = "test_config"
    split = "test_split"
    big_content = {"big": "a" * 100_000_000}
    with pytest.raises(DocumentTooLarge):
        upsert_response(
            kind=kind,
            dataset=dataset,
            dataset_git_revision=dataset_git_revision,
            config=config,
            split=split,
            content=big_content,
            http_status=HTTPStatus.OK,
        )


def test_is_successful_response_two_valid_datasets() -> None:
    kind = CACHE_KIND
    other_kind = "other_kind"
    dataset_a = DATASET_NAME_A
    dataset_git_revision_a = DATASET_GIT_REVISION_A
    dataset_b = DATASET_NAME_B
    dataset_git_revision_b = DATASET_GIT_REVISION_B
    upsert_response(
        kind=kind,
        dataset=dataset_a,
        dataset_git_revision=dataset_git_revision_a,
        content={},
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=kind,
        dataset=dataset_b,
        dataset_git_revision=dataset_git_revision_b,
        content={},
        http_status=HTTPStatus.OK,
    )
    assert is_successful_response(dataset=dataset_a, kind=kind)
    assert is_successful_response(dataset=dataset_b, kind=kind)
    assert not is_successful_response(dataset=dataset_b, kind=other_kind)


def test_count_by_status_and_error_code() -> None:
    assert not get_responses_count_by_kind_status_and_error_code()

    upsert_response(
        kind=CACHE_KIND,
        dataset=DATASET_NAME,
        dataset_git_revision=REVISION_NAME,
        content={"key": "value"},
        http_status=HTTPStatus.OK,
    )

    assert get_responses_count_by_kind_status_and_error_code() == {(CACHE_KIND, 200, None): 1}

    upsert_response(
        kind="test_kind2",
        dataset=DATASET_NAME,
        dataset_git_revision=REVISION_NAME,
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
    assert metrics[(CACHE_KIND, 200, None)] == 1
    assert metrics[("test_kind2", 500, "error_code")] == 1


def test_get_cache_reports() -> None:
    kind = CACHE_KIND
    kind_2 = "test_kind_2"
    expected_cache_reports: CacheReportsPage = {"cache_reports": [], "next_cursor": ""}
    assert get_cache_reports(kind=kind, cursor="", limit=2) == expected_cache_reports
    expected_cache_reports_with_content: CacheReportsWithContentPage = {
        "cache_reports_with_content": [],
        "next_cursor": "",
    }
    assert get_cache_reports_with_content(kind=kind, cursor="", limit=2) == expected_cache_reports_with_content

    dataset_a = DATASET_NAME_A
    dataset_git_revision_a = DATASET_GIT_REVISION_A
    content_a = {"key": "a"}
    http_status_a = HTTPStatus.OK
    updated_at_a = datetime(2020, 1, 1, 0, 0, 0)
    upsert_response(
        kind=kind,
        dataset=dataset_a,
        dataset_git_revision=dataset_git_revision_a,
        content=content_a,
        http_status=http_status_a,
        updated_at=updated_at_a,
    )

    dataset_b = DATASET_NAME_B
    dataset_git_revision_b = DATASET_GIT_REVISION_B
    config_b = "test_config_b"
    content_b = {"key": "b"}
    http_status_b = HTTPStatus.INTERNAL_SERVER_ERROR
    error_code_b = "error_code_b"
    details_b = {
        "error": "error b",
    }
    job_runner_version_b = 0
    updated_at_b = datetime(2020, 1, 1, 0, 0, 1)
    upsert_response(
        kind=kind,
        dataset=dataset_b,
        dataset_git_revision=dataset_git_revision_b,
        config=config_b,
        content=content_b,
        details=details_b,
        http_status=http_status_b,
        error_code=error_code_b,
        job_runner_version=job_runner_version_b,
        updated_at=updated_at_b,
    )

    dataset_c = DATASET_NAME_C
    dataset_git_revision_c = DATASET_GIT_REVISION_C
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
        dataset_git_revision=dataset_git_revision_c,
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
        dataset_git_revision=dataset_git_revision_c,
        content=content_c,
        details=details_c,
        http_status=http_status_c,
        error_code=error_code_c,
        updated_at=updated_at_c,
    )
    upsert_response(
        kind=kind_2,
        dataset=dataset_c,
        dataset_git_revision=dataset_git_revision_c,
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
            "dataset_git_revision": dataset_git_revision_a,
            "progress": None,
            "failed_runs": 0,
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
            "failed_runs": 0,
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
                "dataset_git_revision": dataset_git_revision_c,
                "progress": None,
                "failed_runs": 0,
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
            "dataset_git_revision": dataset_git_revision_a,
            "details": {},
            "updated_at": updated_at_a,
            "progress": None,
            "failed_runs": 0,
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
            "failed_runs": 0,
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
                "dataset_git_revision": dataset_git_revision_c,
                "details": details_c,
                "updated_at": updated_at_c,
                "progress": None,
                "failed_runs": 0,
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
    kind = CACHE_KIND
    content = {"key": "value"}
    http_status = HTTPStatus.OK
    for i in range(num_entries):
        upsert_response(
            kind=kind,
            dataset="dataset",
            dataset_git_revision=REVISION_NAME,
            config="config",
            split=f"split{i}",
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
    kind = CACHE_KIND
    current_version = 2
    minor_version = 1

    result = get_outdated_split_full_names_for_step(kind=kind, current_version=current_version)
    upsert_response(
        kind=kind,
        dataset="dataset_with_current_version",
        dataset_git_revision=REVISION_NAME,
        content={},
        http_status=HTTPStatus.OK,
        job_runner_version=current_version,
    )
    assert not result

    upsert_response(
        kind=kind,
        dataset="dataset_with_minor_version",
        dataset_git_revision=REVISION_NAME,
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
    dataset_git_revision: str
    config: Optional[str]
    http_status: HTTPStatus
    progress: Optional[float]


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
    dataset_git_revision = REVISION_NAME
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
    content={NAMES_FIELD: [{NAME_FIELD: name} for name in NAMES]},
    http_status=HTTPStatus.OK,
)
RESPONSE_ERROR = ResponseSpec(content=CONTENT_ERROR, http_status=HTTPStatus.INTERNAL_SERVER_ERROR)


@pytest.mark.parametrize(
    "response_spec_by_kind,expected_names",
    [
        ({}, []),
        ({CACHE_KIND_A: RESPONSE_ERROR}, []),
        ({CACHE_KIND_A: NAMES_RESPONSE_OK}, NAMES),
        ({CACHE_KIND_B: NAMES_RESPONSE_OK}, []),
    ],
)
def test_fetch_names(
    response_spec_by_kind: Mapping[str, Mapping[str, Any]],
    expected_names: list[str],
) -> None:
    cache_kind = CACHE_KIND_A
    for kind, response_spec in response_spec_by_kind.items():
        upsert_response(
            kind=kind,
            dataset=DATASET_NAME,
            dataset_git_revision=REVISION_NAME,
            config=CONFIG_NAME_1,
            split=None,
            content=response_spec["content"],
            http_status=response_spec["http_status"],
        )
    assert (
        fetch_names(
            dataset=DATASET_NAME,
            config=CONFIG_NAME_1,
            cache_kind=cache_kind,
            names_field=NAMES_FIELD,
            name_field=NAME_FIELD,
        )
        == expected_names
    )


class Entry(TypedDict):
    kind: str
    dataset: str
    dataset_git_revision: str
    config: str
    http_status: HTTPStatus
    updated_at: datetime


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
    "dataset_git_revision": REVISION_A,
}
DATASET_2 = f"{DATASET_NAME}_2"
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
    "dataset_git_revision": REVISION_A,
}
ENTRY_6: Entry = {
    "kind": CACHE_KIND_A,
    "dataset": DATASET_NAME,
    "config": CONFIG_NAME_1,
    "http_status": HTTPStatus.OK,
    "updated_at": BEFORE,
    "dataset_git_revision": REVISION_A,
}
ENTRY_7: Entry = {
    "kind": CACHE_KIND_A,
    "dataset": DATASET_NAME_A,
    "config": CONFIG_NAME_1,
    "http_status": HTTPStatus.OK,
    "updated_at": AFTER_1,
    "dataset_git_revision": REVISION_A,
}


@pytest.mark.parametrize(
    "entries,expected_datasets",
    [
        ([], []),
        ([ENTRY_1], [DATASET_NAME]),
        ([ENTRY_2], []),
        ([ENTRY_4], [DATASET_NAME]),
        ([ENTRY_5], []),
        ([ENTRY_6], []),
        ([ENTRY_1, ENTRY_4], [DATASET_NAME]),
        ([ENTRY_1, ENTRY_2, ENTRY_4, ENTRY_5, ENTRY_6], [DATASET_NAME]),
        ([ENTRY_1, ENTRY_7], [DATASET_NAME, DATASET_NAME_A]),
    ],
)
def test_get_datasets_with_last_updated_kind(entries: list[Entry], expected_datasets: list[str]) -> None:
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
    assert sorted(get_datasets_with_last_updated_kind(kind=kind, days=days)) == sorted(expected_datasets)
    # ^ the order is not meaningful, so we sort to make the test deterministic


def test_get_previous_step_or_raise_success() -> None:
    kind = CACHE_KIND
    dataset = DATASET_NAME
    config = "test_config"
    split = "test_split"
    content = {"key": "value"}

    upsert_response(
        kind=kind,
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        split=split,
        content=content,
        http_status=HTTPStatus.OK,
    )

    try:
        response = get_previous_step_or_raise(kind=kind, dataset=dataset, config=config, split=split)
        assert response["http_status"] == HTTPStatus.OK
        assert response["content"] == content
    finally:
        delete_response(kind=kind, dataset=dataset, config=config, split=split)


def test_get_previous_step_or_raise_not_found() -> None:
    kind = "missing_kind"
    dataset = "missing_dataset"
    config = "missing_config"
    split = "missing_split"

    delete_response(kind=kind, dataset=dataset, config=config, split=split)
    with pytest.raises(CachedArtifactNotFoundError):
        get_previous_step_or_raise(kind=kind, dataset=dataset, config=config, split=split)


def test_get_previous_step_or_raise_error_status() -> None:
    kind = CACHE_KIND
    dataset = "error_dataset"
    config = "error_config"
    split = "error_split"
    content = {"error": "failure"}

    upsert_response(
        kind=kind,
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        split=split,
        content=content,
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        error_code="some_error",
        details={"error": "failure"},
    )

    try:
        with pytest.raises(CachedArtifactError) as exc_info:
            get_previous_step_or_raise(kind=kind, dataset=dataset, config=config, split=split)
        assert exc_info.value.cache_entry_with_details["http_status"] == HTTPStatus.INTERNAL_SERVER_ERROR
        assert exc_info.value.cache_entry_with_details["content"] == content
    finally:
        delete_response(kind=kind, dataset=dataset, config=config, split=split)
