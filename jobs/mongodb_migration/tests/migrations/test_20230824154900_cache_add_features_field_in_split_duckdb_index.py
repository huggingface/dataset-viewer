# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
from typing import Any, Dict, List

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230824154900_cache_add_features_field_in_split_duckdb_index import (
    MigrationAddFeaturesToSplitDuckdbIndexCacheResponse,
)


def assert_features(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert entry["content"]["features"] is None


def assert_unchanged(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert "features" not in entry["content"]


cache: List[Dict[str, Any]] = [
    {
        "config": "lhoestq--demo1",
        "dataset": "lhoestq/demo1",
        "kind": "split-duckdb-index",
        "split": "train",
        "content": {
            "dataset": "lhoestq/demo1",
            "config": "default",
            "split": "train",
            "url": "https://huggingface.co/.../index.duckdb",
            "filename": "index.duckdb",
            "size": 5038,
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
        "kind": "split-duckdb-index",
        "split": "train",
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


def test_cache_add_features(mongo_host: str) -> None:
    with MongoResource(database="test_cache_add_features", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many(cache)

        migration = MigrationAddFeaturesToSplitDuckdbIndexCacheResponse(
            version="20230824154900",
            description="add features field to split-duckdb-index",
        )
        migration.up()

        assert_features("lhoestq/demo1", kind="split-duckdb-index")
        assert_unchanged("lhoestq/error", kind="split-duckdb-index")

        migration.down()
        assert_unchanged("lhoestq/demo1", kind="split-duckdb-index")
        assert_unchanged("lhoestq/error", kind="split-duckdb-index")

        db[CACHE_COLLECTION_RESPONSES].drop()
