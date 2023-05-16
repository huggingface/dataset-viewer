# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Any

from libcommon.queue import Job
from libcommon.simple_cache import CachedResponse
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import CacheMigration, QueueMigration


class CacheRenamingMigration(CacheMigration):
    def __init__(self, cache_kind: str, new_cache_kind: str, *args: Any, **kwargs: Any):
        self.new_cache_kind: str = new_cache_kind
        super().__init__(cache_kind=cache_kind, *args, **kwargs)

    def up(self) -> None:
        logging.info(f"Rename cache_kind field from {self.cache_kind} to {self.new_cache_kind}")
        db = get_db(self.MONGOENGINE_ALIAS)

        # update existing documents with the old kind
        db[self.COLLECTION_RESPONSES].update_many({"kind": self.cache_kind}, {"$set": {"kind": self.new_cache_kind}})

    def down(self) -> None:
        logging.info(f"Rollback cache_kind field from {self.new_cache_kind} to {self.cache_kind}")
        db = get_db(self.MONGOENGINE_ALIAS)
        db[self.COLLECTION_RESPONSES].update_many({"kind": self.new_cache_kind}, {"$set": {"kind": self.cache_kind}})

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=CachedResponse, sample_size=10)


class QueueRenamingMigration(QueueMigration):
    def __init__(self, job_type: str, new_job_type: str, *args: Any, **kwargs: Any):
        self.new_job_type: str = new_job_type
        super().__init__(job_type=job_type, *args, **kwargs)

    def up(self) -> None:
        logging.info(
            f"Rename unicity_id field from Job[{self.job_type}] to "
            f"Job[{self.new_job_type}] and change type from {self.job_type} to "
            f"{self.new_job_type}"
        )

        db = get_db(self.MONGOENGINE_ALIAS)
        db[self.COLLECTION_JOBS].update_many(
            {"type": self.job_type},
            [
                {
                    "$set": {
                        "unicity_id": {
                            "$replaceOne": {
                                "input": "$unicity_id",
                                "find": f"Job[{self.job_type}]",
                                "replacement": f"Job[{self.new_job_type}]",
                            }
                        },
                        "type": self.new_job_type,
                    }
                },
            ],  # type: ignore
        )

    def down(self) -> None:
        logging.info(
            f"Rename unicity_id field from Job[{self.new_job_type}] to "
            f"Job[{self.job_type}] and change type from {self.new_job_type} to "
            f"{self.job_type}"
        )

        db = get_db(self.MONGOENGINE_ALIAS)
        db[self.COLLECTION_JOBS].update_many(
            {"type": self.new_job_type},
            [
                {
                    "$set": {
                        "unicity_id": {
                            "$replaceOne": {
                                "input": "$unicity_id",
                                "find": f"Job[{self.new_job_type}]",
                                "replacement": f"Job[{self.job_type}]",
                            }
                        },
                        "type": self.new_job_type,
                    }
                },
            ],  # type: ignore
        )

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=Job, sample_size=10)
