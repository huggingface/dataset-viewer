# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
from typing import Any

import pytest
from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError
from mongodb_migration.migrations._20240221103200_cache_merge_config_split_names import (
    INFO,
    JOB_RUNNER_VERSION,
    MERGED,
    STREAMING,
    MigrationMergeConfigSplitNamesResponses,
)

DATASET_CONFIG = {"dataset": "dataset", "config": "config"}
ERROR_CODE_OK = "SomeErrorCodeOK"
ERROR_CODE_NOT_OK = "ResponseAlreadyComputedError"
OLD_JOB_RUNNER_VERSION = JOB_RUNNER_VERSION - 1


@pytest.mark.parametrize(
    "entries,expected_entries",
    [
        ([], []),
        # one successful entry: keep it
        ([{"kind": STREAMING, "http_status": 200}], [{"kind": MERGED, "http_status": 200, "content": STREAMING}]),
        ([{"kind": INFO, "http_status": 200}], [{"kind": MERGED, "http_status": 200, "content": INFO}]),
        # one failed entry: keep it (with or without error_code)
        ([{"kind": STREAMING, "http_status": 500}], [{"kind": MERGED, "http_status": 500, "content": STREAMING}]),
        ([{"kind": INFO, "http_status": 500}], [{"kind": MERGED, "http_status": 500, "content": INFO}]),
        (
            [{"kind": STREAMING, "http_status": 500, "error_code": ERROR_CODE_OK, "content": STREAMING}],
            [{"kind": MERGED, "http_status": 500, "error_code": ERROR_CODE_OK, "content": STREAMING}],
        ),
        (
            [{"kind": INFO, "http_status": 500, "error_code": ERROR_CODE_OK, "content": INFO}],
            [{"kind": MERGED, "http_status": 500, "error_code": ERROR_CODE_OK, "content": INFO}],
        ),
        # one "ResponseAlreadyComputedError" entry: remove it
        ([{"kind": STREAMING, "http_status": 500, "error_code": ERROR_CODE_NOT_OK}], []),
        ([{"kind": INFO, "http_status": 500, "error_code": ERROR_CODE_NOT_OK}], []),
        # successful info entry: keep it
        (
            [{"kind": INFO, "http_status": 200}, {"kind": STREAMING, "http_status": 200}],
            [{"kind": MERGED, "http_status": 200, "content": INFO}],
        ),
        (
            [{"kind": INFO, "http_status": 200}, {"kind": STREAMING, "http_status": 500}],
            [{"kind": MERGED, "http_status": 200, "content": INFO}],
        ),
        (
            [
                {"kind": INFO, "http_status": 200},
                {"kind": STREAMING, "http_status": 500, "error_code": ERROR_CODE_NOT_OK},
            ],
            [{"kind": MERGED, "http_status": 200, "content": INFO}],
        ),
        # erroneous info entry: keep the streaming one
        (
            [{"kind": INFO, "http_status": 500}, {"kind": STREAMING, "http_status": 200}],
            [{"kind": MERGED, "http_status": 200, "content": STREAMING}],
        ),
        (
            [{"kind": INFO, "http_status": 500}, {"kind": STREAMING, "http_status": 500}],
            [{"kind": MERGED, "http_status": 500, "content": STREAMING}],
        ),
        (
            [
                {"kind": INFO, "http_status": 500, "error_code": ERROR_CODE_NOT_OK},
                {"kind": STREAMING, "http_status": 500},
            ],
            [{"kind": MERGED, "http_status": 500, "content": STREAMING}],
        ),
        # both "ResponseAlreadyComputedError" entries: remove them
        (
            [
                {"kind": INFO, "http_status": 500, "error_code": ERROR_CODE_NOT_OK},
                {"kind": STREAMING, "http_status": 500, "error_code": ERROR_CODE_NOT_OK},
            ],
            [],
        ),
    ],
)
def test_migration(mongo_host: str, entries: list[dict[str, Any]], expected_entries: list[dict[str, Any]]) -> None:
    with MongoResource(database="test_cache_merge_config_split_names", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)

        if entries:
            db[CACHE_COLLECTION_RESPONSES].insert_many(
                entry | DATASET_CONFIG | {"content": entry["kind"], "job_runner_version": OLD_JOB_RUNNER_VERSION}
                for entry in entries
            )

        migration = MigrationMergeConfigSplitNamesResponses(
            version="20240221103200",
            description="merge 'config-split-names-from-streaming' and 'config-split-names-from-info' responses to 'config-split-names'",
        )
        migration.up()

        assert list(
            {k: v for k, v in entry.items() if k not in ["_id", "dataset", "config"]}
            for entry in db[CACHE_COLLECTION_RESPONSES].find(DATASET_CONFIG)
        ) == list(entry | {"job_runner_version": JOB_RUNNER_VERSION} for entry in expected_entries)

        with pytest.raises(IrreversibleMigrationError):
            migration.down()

        db[CACHE_COLLECTION_RESPONSES].drop()
