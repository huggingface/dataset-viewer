# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
from typing import Any

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230703110100_cache_add_partial_field_in_config_parquet_and_info import (
    MigrationAddPartialToCacheResponse,
)


def assert_partial(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert entry["content"]["partial"] is False


def assert_unchanged(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert "partial" not in entry["content"]


cache: list[dict[str, Any]] = [
    {
        "config": "lhoestq--demo1",
        "dataset": "lhoestq/demo1",
        "kind": "config-parquet-and-info",
        "split": None,
        "content": {
            "parquet_files": [
                {
                    "dataset": "lhoestq/demo1",
                    "config": "lhoestq--demo1",
                    "split": "test",
                    "url": "https://huggingface.co/.../csv-test.parquet",
                    "filename": "csv-test.parquet",
                    "size": 4415,
                },
                {
                    "dataset": "lhoestq/demo1",
                    "config": "lhoestq--demo1",
                    "split": "train",
                    "url": "https://huggingface.co/.../csv-train.parquet",
                    "filename": "csv-train.parquet",
                    "size": 5038,
                },
            ],
            "dataset_info": {
                "description": "",
                "citation": "",
                "homepage": "",
                "license": "",
                "features": {},
                "builder_name": "csv",
                "config_name": "lhoestq--demo1",
                "version": {},
                "splits": {},
                "download_checksums": {},
                "download_size": 2340,
                "dataset_size": 2464,
                "size_in_bytes": 4804,
            },
        },
        "dataset_git_revision": "87ecf163bedca9d80598b528940a9c4f99e14c11",
        "details": None,
        "error_code": None,
        "http_status": 200,
        "job_runner_version": 3,
        "progress": 1.0,
    },
    {
        "config": "lhoestq--error",
        "dataset": "lhoestq/error",
        "kind": "config-parquet-and-info",
        "split": None,
        "content": {"error": "Streaming is not supported for lhoestq/error"},
        "dataset_git_revision": "ec3c8d414af3dfe600399f5e6ef2c682938676f3",
        "details": {
            "error": "Streaming is not supported for lhoestq/error",
            "cause_exception": "TypeError",
            "cause_message": "Streaming is not supported for lhoestq/error",
            "cause_traceback": [
                "Traceback (most recent call last):\n",
                (
                    '  File "/src/services/worker/src/worker/job_manager.py", line 163, in process\n   '
                    " job_result = self.job_runner.compute()\n"
                ),
                (
                    '  File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line'
                    " 932, in compute\n    compute_config_parquet_and_info_response(\n"
                ),
                (
                    '  File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line'
                    " 825, in compute_config_parquet_and_info_response\n    raise_if_not_supported(\n"
                ),
                (
                    '  File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line'
                    " 367, in raise_if_not_supported\n    raise_if_too_big_from_external_data_files(\n"
                ),
                (
                    '  File "/src/services/worker/src/worker/job_runners/config/parquet_and_info.py", line'
                    " 447, in raise_if_too_big_from_external_data_files\n   "
                    " builder._split_generators(mock_dl_manager)\n"
                ),
                (
                    "  File"
                    ' "/tmp/modules-cache/datasets_modules/.../error.py",'
                    ' line 190, in _split_generators\n    raise TypeError("Streaming is not supported for'
                    ' lhoestq/error")\n'
                ),
                "TypeError: Streaming is not supported for lhoestq/error\n",
            ],
        },
        "error_code": "UnexpectedError",
        "http_status": 500,
        "job_runner_version": 3,
        "progress": None,
    },
]
cache2: list[dict[str, Any]] = [
    {
        "config": "lhoestq--demo2",
        "dataset": "lhoestq/demo2",
        "kind": kind,
        "split": None,
        "content": {},
        "http_status": 200,
    }
    for kind in [
        "config-parquet-and-info",
        "config-parquet",
        "dataset-parquet",
        "config-parquet-metadata",
        "config-info",
        "dataset-info",
        "config-size",
        "dataset-size",
    ]
]


def test_cache_add_partial(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_partial", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many(cache + cache2)

        migration = MigrationAddPartialToCacheResponse(
            version="20230703110100",
            description="add partial field to config-parquet-and-info",
        )
        migration.up()

        assert_partial("lhoestq/demo1", kind="config-parquet-and-info")
        assert_unchanged("lhoestq/error", kind="config-parquet-and-info")
        for kind in [
            "config-parquet-and-info",
            "config-parquet",
            "dataset-parquet",
            "config-parquet-metadata",
            "config-info",
            "dataset-info",
            "config-size",
            "dataset-size",
        ]:
            assert_partial("lhoestq/demo2", kind=kind)

        migration.down()
        assert_unchanged("lhoestq/demo1", kind="config-parquet-and-info")
        assert_unchanged("lhoestq/error", kind="config-parquet-and-info")
        for kind in [
            "config-parquet-and-info",
            "config-parquet",
            "dataset-parquet",
            "config-parquet-metadata",
            "config-info",
            "dataset-info",
            "config-size",
            "dataset-size",
        ]:
            assert_unchanged("lhoestq/demo2", kind=kind)

        db[CACHE_COLLECTION_RESPONSES].drop()
