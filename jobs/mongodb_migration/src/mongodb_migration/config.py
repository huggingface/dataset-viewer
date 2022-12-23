# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field

from environs import Env
from libcommon.config import CacheConfig, CommonConfig, QueueConfig

from mongodb_migration.database_migrations import connect_to_database

MONGODB_MIGRATION_MONGO_DATABASE = "datasets_server_maintenance"
MONGO_DATABASE_MONGO_URL = "mongodb://localhost:27017"


@dataclass
class MongodbMigrationConfig:
    mongo_database: str = MONGODB_MIGRATION_MONGO_DATABASE
    mongo_url: str = MONGO_DATABASE_MONGO_URL

    def __post_init__(self):
        connect_to_database(database=self.mongo_database, host=self.mongo_url)

    @staticmethod
    def from_env() -> "MongodbMigrationConfig":
        env = Env(expand_vars=True)
        with env.prefixed("MONGODB_MIGRATION_"):
            return MongodbMigrationConfig(
                mongo_database=env.str(name="MONGO_DATABASE", default=MONGODB_MIGRATION_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=MONGO_DATABASE_MONGO_URL),
            )


@dataclass
class JobConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    mongodb_migration: MongodbMigrationConfig = field(default_factory=MongodbMigrationConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)

    @staticmethod
    def from_env() -> "JobConfig":
        return JobConfig(
            common=CommonConfig.from_env(),
            cache=CacheConfig.from_env(),
            mongodb_migration=MongodbMigrationConfig.from_env(),
            queue=QueueConfig.from_env(),
        )
