from libcommon.resources import MongoResource
from mongoengine.connection import get_db

from mongodb_migration.delete_migration import DeleteCacheMigration

CACHE_MONGOENGINE_ALIAS = DeleteCacheMigration.MONGOENGINE_ALIAS
CACHE_COLLECTION_RESPONSES = DeleteCacheMigration.COLLECTION_RESPONSES


def test_cache_delete_dataset_split_names_from_dataset_info(mongo_host: str) -> None:
    kind = "dataset-split-names-from-dataset-info"
    with MongoResource(
        database="test_cache_delete_dataset_split_names_from_dataset_info", host=mongo_host, mongoengine_alias="cache"
    ):
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        db[CACHE_COLLECTION_RESPONSES].insert_many([{"kind": kind, "dataset": "dataset", "http_status": 200}])
        assert db[CACHE_COLLECTION_RESPONSES].find_one({"kind": kind})  # Ensure there is at least one record to delete

        migration = DeleteCacheMigration(
            cache_kind=kind,
            version="20230504192000",
            description=f"remove cache for kind {kind}",
        )
        migration.up()

        assert not db[CACHE_COLLECTION_RESPONSES].find_one({"kind": kind})  # Ensure 0 records with old kind

        db[CACHE_COLLECTION_RESPONSES].drop()
