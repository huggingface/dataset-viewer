import logging

from libcommon.queue import Job
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

dataset_info = "/dataset-info"
dataset_info_updated = "dataset-info"
db_name = "queue"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationQueueUpdateDatasetInfo(Migration):
    def up(self) -> None:
        logging.info(
            f"Rename unicity_id field from Job[{dataset_info}][<dataset>][<config>][split] to"
            f" Job[{dataset_info_updated}][<dataset>][<config>][split] and change type from {dataset_info} to"
            f" {dataset_info_updated}"
        )

        db = get_db(db_name)
        db["jobsBlue"].update_many(
            {"type": dataset_info},
            [
                {
                    "$set": {
                        "unicity_id": {
                            "$replaceOne": {
                                "input": "$unicity_id",
                                "find": f"Job[{dataset_info}]",
                                "replacement": f"Job[{dataset_info_updated}]",
                            }
                        },
                        "type": dataset_info_updated,
                    }
                },
            ],  # type: ignore
        )

    def down(self) -> None:
        logging.info(
            f"Rename unicity_id field from Job[{dataset_info_updated}][<dataset>][<config>][split] to"
            f" Job[{dataset_info}][<dataset>][<config>][split] and change type from {dataset_info_updated} to"
            f" {dataset_info}"
        )

        db = get_db(db_name)
        db["jobsBlue"].update_many(
            {"type": dataset_info_updated},
            [
                {
                    "$set": {
                        "unicity_id": {
                            "$replaceOne": {
                                "input": "$unicity_id",
                                "find": f"Job[{dataset_info_updated}]",
                                "replacement": f"Job[{dataset_info}]",
                            }
                        },
                        "type": dataset_info_updated,
                    }
                },
            ],  # type: ignore
        )

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=Job, sample_size=10)
