import logging

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import IrreversibleMigrationError, Migration

job_type = "/splits"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationQueueDeleteSplits(Migration):
    def up(self) -> None:
        logging.info(f"Delete jobs of type {job_type}")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].delete_many({"type": job_type})

    def down(self) -> None:
        raise IrreversibleMigrationError("This migration does not support rollback")

    def validate(self) -> None:
        logging.info(f"Check that none of the documents has the {job_type} type")

        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        if db[QUEUE_COLLECTION_JOBS].count_documents({"type": job_type}):
            raise ValueError(f"Found documents with type {job_type}")
