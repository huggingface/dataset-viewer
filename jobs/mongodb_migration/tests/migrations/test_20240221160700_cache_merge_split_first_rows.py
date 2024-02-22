# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
from typing import Any

import pytest
from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError
from mongodb_migration.migrations._20240221160700_cache_merge_split_first_rows import (
    MERGED,
    PARQUET,
    STREAMING,
    MigrationMergeSplitFirstRowsResponses,
)

DATASET_CONFIG_SPLIT = {"dataset": "dataset", "config": "config", "split": "split"}
ERROR_CODE_OK = "SomeErrorCodeOK"
ERROR_CODE_NOT_OK = "ResponseAlreadyComputedError"


@pytest.mark.parametrize(
    "entries,expected_entries",
    [
        ([], []),
        # one successful entry: keep it
        ([{"kind": STREAMING, "http_status": 200}], [{"kind": MERGED, "http_status": 200, "content": STREAMING}]),
        ([{"kind": PARQUET, "http_status": 200}], [{"kind": MERGED, "http_status": 200, "content": PARQUET}]),
        # one failed entry: keep it (with or without error_code)
        ([{"kind": STREAMING, "http_status": 500}], [{"kind": MERGED, "http_status": 500, "content": STREAMING}]),
        ([{"kind": PARQUET, "http_status": 500}], [{"kind": MERGED, "http_status": 500, "content": PARQUET}]),
        (
            [{"kind": STREAMING, "http_status": 500, "error_code": ERROR_CODE_OK, "content": STREAMING}],
            [{"kind": MERGED, "http_status": 500, "error_code": ERROR_CODE_OK, "content": STREAMING}],
        ),
        (
            [{"kind": PARQUET, "http_status": 500, "error_code": ERROR_CODE_OK, "content": PARQUET}],
            [{"kind": MERGED, "http_status": 500, "error_code": ERROR_CODE_OK, "content": PARQUET}],
        ),
        # one "ResponseAlreadyComputedError" entry: remove it
        ([{"kind": STREAMING, "http_status": 500, "error_code": ERROR_CODE_NOT_OK}], []),
        ([{"kind": PARQUET, "http_status": 500, "error_code": ERROR_CODE_NOT_OK}], []),
        # successful parquet entry: keep it
        (
            [{"kind": PARQUET, "http_status": 200}, {"kind": STREAMING, "http_status": 200}],
            [{"kind": MERGED, "http_status": 200, "content": PARQUET}],
        ),
        (
            [{"kind": PARQUET, "http_status": 200}, {"kind": STREAMING, "http_status": 500}],
            [{"kind": MERGED, "http_status": 200, "content": PARQUET}],
        ),
        (
            [
                {"kind": PARQUET, "http_status": 200},
                {"kind": STREAMING, "http_status": 500, "error_code": ERROR_CODE_NOT_OK},
            ],
            [{"kind": MERGED, "http_status": 200, "content": PARQUET}],
        ),
        # erroneous parquet entry: keep the streaming one
        (
            [{"kind": PARQUET, "http_status": 500}, {"kind": STREAMING, "http_status": 200}],
            [{"kind": MERGED, "http_status": 200, "content": STREAMING}],
        ),
        (
            [{"kind": PARQUET, "http_status": 500}, {"kind": STREAMING, "http_status": 500}],
            [{"kind": MERGED, "http_status": 500, "content": STREAMING}],
        ),
        (
            [
                {"kind": PARQUET, "http_status": 500, "error_code": ERROR_CODE_NOT_OK},
                {"kind": STREAMING, "http_status": 500},
            ],
            [{"kind": MERGED, "http_status": 500, "content": STREAMING}],
        ),
        # both "ResponseAlreadyComputedError" entries: remove them
        (
            [
                {"kind": PARQUET, "http_status": 500, "error_code": ERROR_CODE_NOT_OK},
                {"kind": STREAMING, "http_status": 500, "error_code": ERROR_CODE_NOT_OK},
            ],
            [],
        ),
    ],
)
def test_migration(mongo_host: str, entries: list[dict[str, Any]], expected_entries: list[dict[str, Any]]) -> None:
    with MongoResource(database="test_cache_merge_split_first_rows", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)

        if entries:
            db[CACHE_COLLECTION_RESPONSES].insert_many(
                entry | DATASET_CONFIG_SPLIT | {"content": entry["kind"]} for entry in entries
            )

        migration = MigrationMergeSplitFirstRowsResponses(
            version="20240221160700",
            description="merge 'split-first-rows-from-streaming' and 'split-first-rows-from-parquet' responses to 'split-first-rows'",
        )
        migration.up()

        assert (
            list(
                {k: v for k, v in entry.items() if k not in ["_id", "dataset", "config", "split"]}
                for entry in db[CACHE_COLLECTION_RESPONSES].find(DATASET_CONFIG_SPLIT)
            )
            == expected_entries
        )

        with pytest.raises(IrreversibleMigrationError):
            migration.down()

        db[CACHE_COLLECTION_RESPONSES].drop()
