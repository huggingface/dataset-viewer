# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from migration.collector import MigrationsCollector
from migration.config import JobConfig
from migration.plan import Plan

if __name__ == "__main__":
    job_config = JobConfig()
    collected_migrations = MigrationsCollector().get_migrations()
    Plan(collected_migrations=collected_migrations).execute()

# See:
#  https://blog.appsignal.com/2020/04/14/dissecting-rails-migrationsl.html
#  https://edgeguides.rubyonrails.org/active_record_migrations.html
#  https://docs.mongoengine.org/guide/migration.html
#  https://andrewlock.net/deploying-asp-net-core-applications-to-kubernetes-part-7-running-database-migrations/
#  https://helm.sh/docs/topics/charts_hooks/
