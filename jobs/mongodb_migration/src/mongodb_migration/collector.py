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
from mongodb_migration.migrations._20230213170000_split_names_streaming import (
    MigrationUpdateSplitNames,
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
            MigrationUpdateSplitNames(
                version="20230213170000",
                description="update 'kind' field in cache from /split-names to /split-names-streaming",
            ),
        ]
