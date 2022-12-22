# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import sys

from mongodb_migration.collector import MigrationsCollector
from mongodb_migration.config import JobConfig
from mongodb_migration.plan import Plan

if __name__ == "__main__":
    job_config = JobConfig.from_env()
    collected_migrations = MigrationsCollector().get_migrations()
    try:
        Plan(collected_migrations=collected_migrations).execute()
        sys.exit(0)
    except Exception:
        sys.exit(1)

# See:
#  https://blog.appsignal.com/2020/04/14/dissecting-rails-migrationsl.html
#  https://edgeguides.rubyonrails.org/active_record_migrations.html
#  https://docs.mongoengine.org/guide/migration.html
#  https://andrewlock.net/deploying-asp-net-core-applications-to-kubernetes-part-7-running-database-migrations/
#  https://helm.sh/docs/topics/charts_hooks/
