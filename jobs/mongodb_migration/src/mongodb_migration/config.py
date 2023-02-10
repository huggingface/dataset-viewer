# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field

from environs import Env
from libcommon.config import CacheConfig, CommonConfig, QueueConfig

DATABASE_MIGRATIONS_MONGO_DATABASE = "datasets_server_maintenance"
DATABASE_MIGRATIONS_MONGO_URL = "mongodb://localhost:27017"


@dataclass(frozen=True)
class DatabaseMigrationsConfig:
    mongo_database: str = DATABASE_MIGRATIONS_MONGO_DATABASE
    mongo_url: str = DATABASE_MIGRATIONS_MONGO_URL

    @staticmethod
    def from_env() -> "DatabaseMigrationsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("DATABASE_MIGRATIONS_"):
            return DatabaseMigrationsConfig(
                mongo_database=env.str(name="MONGO_DATABASE", default=DATABASE_MIGRATIONS_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=DATABASE_MIGRATIONS_MONGO_URL),
            )


@dataclass(frozen=True)
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
