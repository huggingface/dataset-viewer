# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List

from mongodb_migration.migration import Migration
from mongodb_migration.migrations._20221110230400_example import MigrationExample
from mongodb_migration.migrations._20221116133500_queue_job_add_force import (
    MigrationAddForceToJob,
)
from mongodb_migration.migrations._20221117223000_cache_generic_response import (
    MigrationMoveToGenericCachedResponse,
)
from mongodb_migration.migrations._20230126164900_queue_job_add_priority import (
    MigrationAddPriorityToJob,
)
from mongodb_migration.migrations._20230216112500_cache_split_names_from_streaming import (
    MigrationCacheUpdateSplitNames,
)
from mongodb_migration.migrations._20230216141000_queue_split_names_from_streaming import (
    MigrationQueueUpdateSplitNames,
)
from mongodb_migration.migrations._20230309123100_cache_add_progress import (
    MigrationAddProgressToCacheResponse,
)
from mongodb_migration.migrations._20230309141600_cache_add_job_runner_version import (
    MigrationAddJobRunnerVersionToCacheResponse,
)


# TODO: add a way to automatically collect migrations from the migrations/ folder
class MigrationsCollector:
    def get_migrations(self) -> List[Migration]:
        return [
            MigrationExample(version="20221110230400", description="example"),
            MigrationAddForceToJob(
                version="20221116133500", description="add 'force' field to jobs in queue database"
            ),
            MigrationMoveToGenericCachedResponse(
                version="20221117223000",
                description="replace SplitsResponse and FirstRowsResponse with a generic CachedResponse",
            ),
            MigrationAddPriorityToJob(
                version="20230126164900",
                description="add 'priority' field to jobs in queue database",
            ),
            MigrationCacheUpdateSplitNames(
                version="20230216112500",
                description="update 'kind' field in cache from /split-names to /split-names-from-streaming",
            ),
            MigrationQueueUpdateSplitNames(
                version="20230216141000",
                description=(
                    "update 'type' and 'unicity_id' fields in job from /split-names to /split-names-from-streaming"
                ),
            ),
            MigrationAddProgressToCacheResponse(
                version="20230309123100",
                description="add the 'progress' field with the default value (1.0) to the cached results",
            ),
            MigrationAddJobRunnerVerionToCacheResponse(
                version="20230309141600", description="add 'job_runner_version' field based on 'worker_version' value"
            ),
        ]
