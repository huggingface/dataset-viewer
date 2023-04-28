# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from libcommon.queue import Job
from libcommon.utils import get_datetime
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230428145000_queue_delete_ttl_index import (
    MigrationQueueDeleteTTLIndexOnFinishedAt,
    get_index_names,
)


def test_queue_delete_ttl_index(mongo_host: str) -> None:
    with MongoResource(database="test_queue_delete_ttl_index", host=mongo_host, mongoengine_alias="queue"):
        Job(type="test", dataset="test", unicity_id="test", namespace="test", created_at=get_datetime()).save()
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        assert (
            len(get_index_names(db[QUEUE_COLLECTION_JOBS].index_information(), "finished_at")) == 1
        )  # Ensure the TTL index exists

        migration = MigrationQueueDeleteTTLIndexOnFinishedAt(
            version="20230428145000",
            description="remove ttl index on field 'finished_at'",
        )
        migration.up()

        assert (
            len(get_index_names(db[QUEUE_COLLECTION_JOBS].index_information(), "finished_at")) == 0
        )  # Ensure the TTL index exists  # Ensure 0 records with old type

        db[QUEUE_COLLECTION_JOBS].drop()
