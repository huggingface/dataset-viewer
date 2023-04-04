# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import sys

from libcommon.log import init_logging
from libcommon.resources import CacheMongoResource, QueueMongoResource

from mongodb_migration.collector import MigrationsCollector
from mongodb_migration.config import JobConfig
from mongodb_migration.plan import Plan
from mongodb_migration.resources import MigrationsMongoResource


def run_job() -> None:
    job_config = JobConfig.from_env()

    init_logging(level=job_config.log.level)
    # ^ set first to have logs as soon as possible

    with (
        CacheMongoResource(
            database=job_config.cache.mongo_database, host=job_config.cache.mongo_url
        ) as cache_resource,
        QueueMongoResource(
            database=job_config.queue.mongo_database, host=job_config.queue.mongo_url
        ) as queue_resource,
        MigrationsMongoResource(
            database=job_config.database_migrations.mongo_database, host=job_config.database_migrations.mongo_url
        ) as migrations_database_resource,
    ):
        if not cache_resource.is_available():
            logging.warning(
                "The connection to the cache database could not be established. The migration job is skipped."
            )
            return
        if not queue_resource.is_available():
            logging.warning(
                "The connection to the queue database could not be established. The migration job is skipped."
            )
            return
        if not migrations_database_resource.is_available():
            logging.warning(
                "The connection to the migrations database could not be established. The migration job is skipped."
            )
            return
        collected_migrations = MigrationsCollector().get_migrations()
        Plan(collected_migrations=collected_migrations).execute()


if __name__ == "__main__":
    try:
        run_job()
        sys.exit(0)
    except Exception:
        sys.exit(1)

# See:
#  https://blog.appsignal.com/2020/04/14/dissecting-rails-migrationsl.html
#  https://edgeguides.rubyonrails.org/active_record_migrations.html
#  https://docs.mongoengine.org/guide/migration.html
#  https://andrewlock.net/deploying-asp-net-core-applications-to-kubernetes-part-7-running-database-migrations/
#  https://helm.sh/docs/topics/charts_hooks/
