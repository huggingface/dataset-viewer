# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.constants import QUEUE_COLLECTION_JOBS, QUEUE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.migration import Migration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationAddForceToJob(Migration):
    def up(self) -> None:
        # See https://docs.mongoengine.org/guide/migration.html#example-1-addition-of-a-field
        logging.info("If missing, add the force field with the default value (False) to  the jobs")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many({"force": {"$exists": False}}, {"$set": {"force": False}})

    def down(self) -> None:
        logging.info("Remove the force field from all the jobs")
        db = get_db(QUEUE_MONGOENGINE_ALIAS)
        db[QUEUE_COLLECTION_JOBS].update_many({}, {"$unset": {"force": ""}})

    def validate(self) -> None:
        logging.info("Ensure that a random selection of jobs have the 'force' field")

        # The Job object does not contain the force field anymore. See _20230511100600_queue_remove_force.py
        # check_documents(DocCls=Job, sample_size=10)
