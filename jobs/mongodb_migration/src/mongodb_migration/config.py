# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field

from environs import Env
from libcommon.config import CacheConfig, CommonConfig, QueueConfig

MONGODB_MIGRATION_MONGO_DATABASE = "datasets_server_maintenance"
MONGODB_MIGRATION_MONGO_URL = "mongodb://localhost:27017"


@dataclass
class DatabaseMigrationsConfig:
    mongo_database: str = MONGODB_MIGRATION_MONGO_DATABASE
    mongo_url: str = MONGODB_MIGRATION_MONGO_URL

    @staticmethod
    def from_env() -> "DatabaseMigrationsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("MONGODB_MIGRATION_"):
            return DatabaseMigrationsConfig(
                mongo_database=env.str(name="MONGO_DATABASE", default=MONGODB_MIGRATION_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=MONGODB_MIGRATION_MONGO_URL),
            )


@dataclass
class JobConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    common: CommonConfig = field(default_factory=CommonConfig)
    database_migrations: DatabaseMigrationsConfig = field(default_factory=DatabaseMigrationsConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)

    @staticmethod
    def from_env() -> "JobConfig":
        return JobConfig(
            common=CommonConfig.from_env(),
            cache=CacheConfig.from_env(),
            database_migrations=DatabaseMigrationsConfig.from_env(),
            queue=QueueConfig.from_env(),
        )
