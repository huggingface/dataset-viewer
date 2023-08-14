# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from dataclasses import dataclass, field

from environs import Env
from libcommon.config import CacheConfig, LogConfig, QueueConfig

# TODO: Remove this config once all the collections in metrics db have been removed
METRICS_MONGO_DATABASE = "datasets_server_metrics"
METRICS_MONGO_URL = "mongodb://localhost:27017"


@dataclass(frozen=True)
class MetricsConfig:
    mongo_database: str = METRICS_MONGO_DATABASE
    mongo_url: str = METRICS_MONGO_URL

    @classmethod
    def from_env(cls) -> "MetricsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("METRICS_"):
            return cls(
                mongo_database=env.str(name="MONGO_DATABASE", default=METRICS_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=METRICS_MONGO_URL),
            )


DATABASE_MIGRATIONS_MONGO_DATABASE = "datasets_server_maintenance"
DATABASE_MIGRATIONS_MONGO_URL = "mongodb://localhost:27017"


@dataclass(frozen=True)
class DatabaseMigrationsConfig:
    mongo_database: str = DATABASE_MIGRATIONS_MONGO_DATABASE
    mongo_url: str = DATABASE_MIGRATIONS_MONGO_URL

    @classmethod
    def from_env(cls) -> "DatabaseMigrationsConfig":
        env = Env(expand_vars=True)
        with env.prefixed("DATABASE_MIGRATIONS_"):
            return cls(
                mongo_database=env.str(name="MONGO_DATABASE", default=DATABASE_MIGRATIONS_MONGO_DATABASE),
                mongo_url=env.str(name="MONGO_URL", default=DATABASE_MIGRATIONS_MONGO_URL),
            )


@dataclass(frozen=True)
class JobConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    log: LogConfig = field(default_factory=LogConfig)
    database_migrations: DatabaseMigrationsConfig = field(default_factory=DatabaseMigrationsConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)

    @classmethod
    def from_env(cls) -> "JobConfig":
        return cls(
            log=LogConfig.from_env(),
            cache=CacheConfig.from_env(),
            database_migrations=DatabaseMigrationsConfig.from_env(),
            metrics=MetricsConfig.from_env(),
            queue=QueueConfig.from_env(),
        )
