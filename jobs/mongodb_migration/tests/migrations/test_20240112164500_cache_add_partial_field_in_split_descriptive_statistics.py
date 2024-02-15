# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
from typing import Any

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.dtos import SplitHubFile
from libcommon.parquet_utils import parquet_export_is_partial
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20240112164500_cache_add_partial_field_in_split_descriptive_statistics import (
    MigrationAddPartialToSplitDescriptiveStatisticsCacheResponse,
)


def assert_partial_field(dataset: str, split: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "split": split, "kind": kind})
    assert entry is not None

    entry_parquet = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": "config-parquet"})
    partial = parquet_export_is_partial(
        [file for file in entry_parquet["content"]["parquet_files"] if file["split"] == split][0]["url"]  # type: ignore
    )
    assert "partial" in entry["content"]
    assert entry["content"]["partial"] is partial


def assert_unchanged(dataset: str, kind: str) -> None:
    db = get_db(CACHE_MONGOENGINE_ALIAS)
    entry = db[CACHE_COLLECTION_RESPONSES].find_one({"dataset": dataset, "kind": kind})
    assert entry is not None
    assert "partial" not in entry["content"]


partial_parquet_files = [
    SplitHubFile(
        dataset="dataset_partial",
        config="default",
        split="train",
        url="https://huggingface.co/datasets/dummy/test/resolve/refs%2Fconvert%2Fparquet/default/partial-train/0000.parquet",
        filename="0000.parquet",
        size=2,
    ),
    SplitHubFile(
        dataset="dataset_partial",
        config="default",
        split="test",
        url="https://huggingface.co/datasets/dummy/test/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet",
        filename="0000.parquet",
        size=1,
    ),
]


def test_cache_add_partial(mongo_host: str) -> None:
    kind = "split-descriptive-statistics"
    with MongoResource(database="test_cache_add_partial", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        cache: list[dict[str, Any]] = [
            {
                "config": "default",
                "dataset": "dataset",
                "kind": kind,
                "split": "train",
                "content": {
                    "num_examples": 20,
                    "statistics": {},
                },
                "http_status": 200,
                "job_runner_version": 3,
                "progress": 1,
            },
            {
                "config": "default",
                "dataset": "dataset_partial",
                "kind": kind,
                "split": "train",
                "content": {
                    "num_examples": 20,
                    "statistics": {},
                },
                "http_status": 200,
                "job_runner_version": 3,
                "progress": 1,
            },
            {
                "config": "default",
                "dataset": "dataset_partial",
                "kind": kind,
                "split": "test",  # check multiple splits because config-parquet is config-level
                "content": {
                    "num_examples": 20,
                    "statistics": {},
                },
                "http_status": 200,
                "job_runner_version": 3,
                "progress": 1,
            },
            {
                "config": "default",
                "dataset": "dataset_with_error",
                "kind": kind,
                "split": "train",
                "content": {"error": "error"},
                "details": {
                    "error": "error",
                    "cause_exception": "SplitWithTooBigParquetError",
                    "cause_message": "error",
                    "cause_traceback": ["Traceback"],
                },
                "error_code": "SplitWithTooBigParquetError",
                "http_status": 500,
                "job_runner_version": 3,
                "progress": 1,
            },
            {
                "config": "default",
                "dataset": "dataset",
                "kind": "config-parquet",
                "split": "train",
                "content": {
                    "parquet_files": [
                        SplitHubFile(
                            dataset="dataset",
                            config="default",
                            split="train",
                            url="https://huggingface.co/datasets/dummy/test/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet",
                            filename="0000.parquet",
                            size=2,
                        ),
                    ],
                    "features": {},
                    "partial": False,
                },
                "http_status": 200,
                "job_runner_version": 3,
                "progress": 1,
            },
            {
                "config": "default",
                "dataset": "dataset_partial",
                "kind": "config-parquet",
                "split": None,
                "content": {"parquet_files": partial_parquet_files, "features": {}, "partial": True},
                "http_status": 200,
                "job_runner_version": 3,
                "progress": 1,
            },
        ]

        db[CACHE_COLLECTION_RESPONSES].insert_many(cache)

        migration = MigrationAddPartialToSplitDescriptiveStatisticsCacheResponse(
            version="20240112164500",
            description="add 'partial' fields for 'split-descriptive-statistics' cache records. ",
        )
        migration.up()

        assert_partial_field("dataset", "train", kind=kind)
        assert_partial_field("dataset_partial", "train", kind=kind)
        assert_partial_field("dataset_partial", "test", kind=kind)
        assert_unchanged("dataset_with_error", kind=kind)

        migration.down()
        assert_unchanged("dataset", kind=kind)
        assert_unchanged("dataset_partial", kind=kind)
        assert_unchanged("dataset_with_error", kind=kind)

        db[CACHE_COLLECTION_RESPONSES].drop()
