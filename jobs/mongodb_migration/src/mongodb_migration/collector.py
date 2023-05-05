# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List

from mongodb_migration.deletion_migrations import (
    CacheDeletionMigration,
    MetricsDeletionMigration,
    QueueDeletionMigration,
)
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
from mongodb_migration.migrations._20230313164200_cache_remove_worker_version import (
    MigrationRemoveWorkerVersionFromCachedResponse,
)
from mongodb_migration.migrations._20230320163700_cache_first_rows_from_streaming import (
    MigrationCacheUpdateFirstRows,
)
from mongodb_migration.migrations._20230320165700_queue_first_rows_from_streaming import (
    MigrationQueueUpdateFirstRows,
)
from mongodb_migration.migrations._20230323155000_cache_dataset_info import (
    MigrationCacheUpdateDatasetInfo,
)
from mongodb_migration.migrations._20230323160000_queue_dataset_info import (
    MigrationQueueUpdateDatasetInfo,
)
from mongodb_migration.migrations._20230428145000_queue_delete_ttl_index import (
    MigrationQueueDeleteTTLIndexOnFinishedAt,
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
            MigrationAddJobRunnerVersionToCacheResponse(
                version="20230309141600", description="add 'job_runner_version' field based on 'worker_version' value"
            ),
            MigrationRemoveWorkerVersionFromCachedResponse(
                version="20230313164200", description="remove 'worker_version' field from cache"
            ),
            MigrationCacheUpdateFirstRows(
                version="20230320163700",
                description="update 'kind' field in cache from /first-rows to split-first-rows-from-streaming",
            ),
            MigrationQueueUpdateFirstRows(
                version="20230320165700",
                description=(
                    "update 'type' and 'unicity_id' fields in job from /first-rows to split-first-rows-from-streaming"
                ),
            ),
            MigrationCacheUpdateDatasetInfo(
                version="20230323155000",
                description="update 'kind' field in cache from '/dataset-info' to 'dataset-info'",
            ),
            MigrationQueueUpdateDatasetInfo(
                version="20230323160000",
                description="update 'type' and 'unicity_id' fields in job from /dataset-info to dataset-info",
            ),
            QueueDeletionMigration(
                job_type="/splits",
                version="20230407091400",
                description="delete the jobs of type '/splits'",
            ),
            CacheDeletionMigration(
                cache_kind="/splits",
                version="20230407091500",
                description="delete the cache entries of kind '/splits'",
            ),
            QueueDeletionMigration(
                job_type="/parquet-and-dataset-info",
                version="20230424173000",
                description="delete the jobs of type '/parquet-and-dataset-info'",
            ),
            CacheDeletionMigration(
                cache_kind="/parquet-and-dataset-info",
                version="20230424174000",
                description="delete the cache entries of kind '/parquet-and-dataset-info'",
            ),
            MetricsDeletionMigration(
                job_type="/parquet-and-dataset-info",
                cache_kind="'/parquet-and-dataset-info'",
                version="20230427121500",
                description="delete the queue and cache metrics for step '/parquet-and-dataset-info'",
            ),
            MigrationQueueDeleteTTLIndexOnFinishedAt(
                version="20230428145000",
                description="delete the TTL index on the 'finished_at' field in the queue database",
            ),
            CacheDeletionMigration(
                cache_kind="dataset-split-names-from-streaming",
                version="20230428175100",
                description="delete the cache entries of kind 'dataset-split-names-from-streaming'",
            ),
            QueueDeletionMigration(
                job_type="dataset-split-names-from-streaming",
                version="20230428181800",
                description="delete the jobs of type 'dataset-split-names-from-streaming'",
            ),
            MetricsDeletionMigration(
                job_type="dataset-split-names-from-streaming",
                cache_kind="dataset-split-names-from-streaming",
                version="20230428193100",
                description="delete the queue and cache metrics for step 'dataset-split-names-from-streaming'",
            ),
            CacheDeletionMigration(
                cache_kind="dataset-split-names-from-dataset-info",
                version="20230504185100",
                description="delete the cache entries of kind 'dataset-split-names-from-dataset-info'",
            ),
            QueueDeletionMigration(
                job_type="dataset-split-names-from-dataset-info",
                version="20230504192200",
                description="delete the jobs of type 'dataset-split-names-from-dataset-info'",
            ),
            MetricsDeletionMigration(
                job_type="dataset-split-names-from-dataset-info",
                cache_kind="dataset-split-names-from-dataset-info",
                version="20230504194600",
                description="delete the queue and cache metrics for step 'dataset-split-names-from-dataset-info'",
            ),
        ]
