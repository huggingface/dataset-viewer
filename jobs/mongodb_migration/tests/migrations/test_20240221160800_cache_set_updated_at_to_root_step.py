# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
from typing import Any

import pytest
from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError
from mongodb_migration.migrations._20240221160800_cache_set_updated_at_to_root_step import (
    ROOT_STEP,
    MigrationSetUpdatedAtToOldestStep,
)

STEP_1 = "step-1"
STEP_2 = "step-2"

DATE_0 = "2024-02-01T00:00:00.000000Z"
DATE_1 = "2024-02-02T00:00:00.000000Z"
DATE_2 = "2024-02-03T00:00:00.000000Z"

DATASET_1 = "dataset1"
DATASET_2 = "dataset2"


@pytest.mark.parametrize(
    "entries,expected_entries",
    [
        ([], []),
        (
            [{"kind": ROOT_STEP, "dataset": DATASET_1, "updated_at": DATE_1}],
            [{"kind": ROOT_STEP, "dataset": DATASET_1, "updated_at": DATE_1}],
        ),
        (
            [
                {"kind": ROOT_STEP, "dataset": DATASET_1, "updated_at": DATE_1},
                {"kind": STEP_1, "dataset": DATASET_1, "updated_at": DATE_2},
                {"kind": STEP_2, "dataset": DATASET_1, "updated_at": DATE_0},
            ],
            [
                {"kind": ROOT_STEP, "dataset": DATASET_1, "updated_at": DATE_1},
                {"kind": STEP_1, "dataset": DATASET_1, "updated_at": DATE_1},
                {"kind": STEP_2, "dataset": DATASET_1, "updated_at": DATE_1},
            ],
        ),
        (
            [
                {"kind": ROOT_STEP, "dataset": DATASET_1, "updated_at": DATE_1},
                {"kind": STEP_1, "dataset": DATASET_1, "updated_at": DATE_2},
                {"kind": ROOT_STEP, "dataset": DATASET_2, "updated_at": DATE_2},
                {"kind": STEP_1, "dataset": DATASET_2, "updated_at": DATE_0},
            ],
            [
                {"kind": ROOT_STEP, "dataset": DATASET_1, "updated_at": DATE_1},
                {"kind": STEP_1, "dataset": DATASET_1, "updated_at": DATE_1},
                {"kind": ROOT_STEP, "dataset": DATASET_2, "updated_at": DATE_2},
                {"kind": STEP_1, "dataset": DATASET_2, "updated_at": DATE_2},
            ],
        ),
    ],
)
def test_migration(mongo_host: str, entries: list[dict[str, Any]], expected_entries: list[dict[str, Any]]) -> None:
    with MongoResource(database="test_cache_set_updated_at_to_root_step", host=mongo_host, mongoengine_alias="cache"):
        db = get_db(CACHE_MONGOENGINE_ALIAS)

        if entries:
            db[CACHE_COLLECTION_RESPONSES].insert_many(entries)

        migration = MigrationSetUpdatedAtToOldestStep(
            version="20240221160800",
            description="set 'updated_at' of the root step to all the cache entries for each dataset",
        )
        migration.up()

        assert (
            list({k: v for k, v in entry.items() if k != "_id"} for entry in db[CACHE_COLLECTION_RESPONSES].find())
            == expected_entries
        )

        with pytest.raises(IrreversibleMigrationError):
            migration.down()

        db[CACHE_COLLECTION_RESPONSES].drop()
