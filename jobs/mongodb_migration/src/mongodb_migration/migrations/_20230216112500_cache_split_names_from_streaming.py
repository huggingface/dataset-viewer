# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.constants import CACHE_COLLECTION_RESPONSES, CACHE_MONGOENGINE_ALIAS
from mongoengine.connection import get_db

from mongodb_migration.renaming_migrations import CacheRenamingMigration


# connection already occurred in the main.py (caveat: we use globals)
class MigrationCacheUpdateSplitNames(CacheRenamingMigration):
    def up(self) -> None:
        tmp_kind = f"{self.cache_kind}-TMP"
        logging.info(f"Rename cache_kind field from {self.cache_kind} to {self.new_cache_kind}")
        db = get_db(CACHE_MONGOENGINE_ALIAS)
        # update existing documents with the new kind (if any) to avoid duplicates (will be deleted later)
        db[CACHE_COLLECTION_RESPONSES].update_many({"kind": self.new_cache_kind}, {"$set": {"kind": tmp_kind}})
        # update existing documents with the old kind
        db[CACHE_COLLECTION_RESPONSES].update_many({"kind": self.cache_kind}, {"$set": {"kind": self.new_cache_kind}})
        # delete the duplicates
        db[CACHE_COLLECTION_RESPONSES].delete_many({"kind": tmp_kind})
