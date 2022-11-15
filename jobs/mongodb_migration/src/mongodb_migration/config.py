# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from environs import Env
from libcache.config import CacheConfig
from libcommon.config import CommonConfig
from libqueue.config import QueueConfig

from mongodb_migration.database_migrations import connect_to_database


class MongodbMigrationConfig:
    mongo_database: str
    mongo_url: str

    def __init__(self):
        env = Env(expand_vars=True)
        with env.prefixed("MONGODB_MIGRATION_"):
            self.mongo_database = env.str(name="MONGO_DATABASE", default="datasets_server_maintenance")
            self.mongo_url = env.str(name="MONGO_URL", default="mongodb://localhost:27017")
        self.setup()

    def setup(self):
        connect_to_database(database=self.mongo_database, host=self.mongo_url)


class JobConfig:
    cache: CacheConfig
    common: CommonConfig
    mongodb_migration: MongodbMigrationConfig
    queue: QueueConfig

    def __init__(self):
        # First process the common configuration to setup the logging
        self.common = CommonConfig()
        self.cache = CacheConfig()
        self.mongodb_migration = MongodbMigrationConfig()
        self.queue = QueueConfig()
