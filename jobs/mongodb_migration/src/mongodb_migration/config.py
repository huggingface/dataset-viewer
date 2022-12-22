# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass

from environs import Env
from libcommon.config import CacheConfig, CommonConfig, QueueConfig

from mongodb_migration.database_migrations import connect_to_database


@dataclass
class MongodbMigrationConfig:
    mongo_database: str = "datasets_server_maintenance"
    mongo_url: str = "mongodb://localhost:27017"

    def __post_init__(self):
        connect_to_database(database=self.mongo_database, host=self.mongo_url)

    @staticmethod
    def from_env() -> "MongodbMigrationConfig":
        env = Env(expand_vars=True)
        with env.prefixed("MONGODB_MIGRATION_"):
            return MongodbMigrationConfig(
                mongo_database=env.str(name="MONGO_DATABASE", default=None),
                mongo_url=env.str(name="MONGO_URL", default=None),
            )


@dataclass
class JobConfig:
    cache: CacheConfig
    common: CommonConfig
    mongodb_migration: MongodbMigrationConfig
    queue: QueueConfig

    @staticmethod
    def from_env() -> "JobConfig":
        return JobConfig(
            common=CommonConfig.from_env(),
            cache=CacheConfig.from_env(),
            mongodb_migration=MongodbMigrationConfig.from_env(),
            queue=QueueConfig.from_env(),
        )
