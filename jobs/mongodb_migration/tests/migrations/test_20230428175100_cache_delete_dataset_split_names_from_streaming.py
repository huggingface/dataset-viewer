from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.migrations._20230428175100_cache_delete_dataset_split_names_from_streaming import (
    MigrationCacheDeleteDatasetSplitNamesFromStreaming,
)


def test_cache_delete_dataset_split_names_from_streaming(mongo_host: str) -> None:
    kind = "dataset-split-names-from-streaming"
    with MongoResource(
        database="test_cache_delete_dataset_split_names_from_streaming", host=mongo_host, mongoengine_alias="cache"
    ):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many([{"kind": kind, "dataset": "dataset", "http_status": 200}])
        assert db[CACHE_COLLECTION_RESPONSES].find_one({"kind": kind})  # Ensure there is at least one record to delete

        migration = MigrationCacheDeleteDatasetSplitNamesFromStreaming(
            version="20230428180400",
            description=f"remove cache for kind {kind}",
        )
        migration.up()

        assert not db[CACHE_COLLECTION_RESPONSES].find_one({"kind": kind})  # Ensure 0 records with old kind

        db[CACHE_COLLECTION_RESPONSES].drop()
