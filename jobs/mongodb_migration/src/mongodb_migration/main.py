# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import sys

from mongodb_migration.collector import MigrationsCollector
from mongodb_migration.config import CheckError, JobConfig
from mongodb_migration.plan import Plan


def run_job() -> None:
    try:
        job_config = JobConfig.from_env()
    except CheckError:
        logging.info("No connection to the database, skipping migrations")
        return
    collected_migrations = MigrationsCollector().get_migrations()
    Plan(collected_migrations=collected_migrations).execute()

    # TODO: use a context manager
    job_config.mongodb_migration.mongo_connection.disconnect()
    job_config.cache.mongo_connection.disconnect()
    job_config.queue.mongo_connection.disconnect()


if __name__ == "__main__":
    try:
        run_job()
        sys.exit(0)
    except Exception as e:
        logging.error(e, exc_info=True)
        sys.exit(1)

# See:
#  https://blog.appsignal.com/2020/04/14/dissecting-rails-migrationsl.html
#  https://edgeguides.rubyonrails.org/active_record_migrations.html
#  https://docs.mongoengine.org/guide/migration.html
#  https://andrewlock.net/deploying-asp-net-core-applications-to-kubernetes-part-7-running-database-migrations/
#  https://helm.sh/docs/topics/charts_hooks/
