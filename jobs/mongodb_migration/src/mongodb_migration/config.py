# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field

from environs import Env
from libcommon.config import CacheConfig, CommonConfig, QueueConfig
from libcommon.mongo import CheckError, MongoConnection

from mongodb_migration.database_migrations import MAINTENANCE_DATABASE_ALIAS

MONGODB_MIGRATION_MONGO_CONNECTION_TIMEOUT_MS = 30_000
MONGODB_MIGRATION_MONGO_DATABASE = "datasets_server_maintenance"
MONGODB_MIGRATION_MONGO_URL = "mongodb://localhost:27017"


@dataclass
class MongodbMigrationConfig:
    mongo_connection_timeout_ms: int = MONGODB_MIGRATION_MONGO_CONNECTION_TIMEOUT_MS
    mongo_database: str = MONGODB_MIGRATION_MONGO_DATABASE
    mongo_url: str = MONGODB_MIGRATION_MONGO_URL

    mongo_connection: MongoConnection = field(init=False)

    def __post_init__(self):
        self.mongo_connection = MongoConnection(
            database=self.mongo_database,
            alias=MAINTENANCE_DATABASE_ALIAS,
            host=self.mongo_url,
            serverSelectionTimeoutMS=self.mongo_connection_timeout_ms,
        )
        self.mongo_connection.check_connection()

    @staticmethod
    def from_env() -> "MongodbMigrationConfig":
        env = Env(expand_vars=True)
        with env.prefixed("MONGODB_MIGRATION_"):
            return MongodbMigrationConfig(
                mongo_connection_timeout_ms=env.str(
                    name="MONGO_CONNECTION_TIMEOUT_MS", default=MONGODB_MIGRATION_MONGO_CONNECTION_TIMEOUT_MS
                ),
                mongo_database=env.str(name="MONGO_DATABASE", default=MONGODB_MIGRATION_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=MONGODB_MIGRATION_MONGO_URL),
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


# explicit re-export
__all__ = ["CheckError"]
