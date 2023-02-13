# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.queue import Job
from libcommon.simple_cache import CachedResponse
from mongoengine.connection import get_db

from mongodb_migration.check import check_documents
from mongodb_migration.migration import Migration

db_name = "cache"


# connection already occurred in the main.py (caveat: we use globals)
class MigrationUpdateSplitNames(Migration):
    def up(self) -> None:
        logging.info(f"Rename {db_name} kind field from /split-name to /split-names-streaming")
        db = get_db(db_name)
        db["cachedResponsesBlue"].update_many({"kind": "/split-names"}, {"$set": {"kind": "/split-names-streaming"}})

    def down(self) -> None:
        logging.info(f"Rollback {db_name} kind field from /split-name-streaming to /split-names")
        db = get_db(db_name)
        db["cachedResponsesBlue"].update_many({"kind": "/split-names-streaming"}, {"$set": {"kind": "/split-names"}})

    def validate(self) -> None:
        logging.info("Validate modified documents")

        check_documents(DocCls=Job, sample_size=10)

        cached_responses_count = CachedResponse.objects(kind="/split-names").count()
        if cached_responses_count > 0:
            raise ValueError(f"{cached_responses_count} documents were not migrated.")
