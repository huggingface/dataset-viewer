# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.constants import (
    CACHE_METRICS_COLLECTION,
    METRICS_MONGOENGINE_ALIAS,
    QUEUE_METRICS_COLLECTION,
)

from mongodb_migration.deletion_migrations import (
    CacheDeletionMigration,
    MetricsDeletionMigration,
    MigrationQueueDeleteTTLIndex,
    QueueDeletionMigration,
)
from mongodb_migration.drop_migrations import MigrationDropCollection
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
from mongodb_migration.migrations._20230309123100_cache_add_progress import (
    MigrationAddProgressToCacheResponse,
)
from mongodb_migration.migrations._20230309141600_cache_add_job_runner_version import (
    MigrationAddJobRunnerVersionToCacheResponse,
)
from mongodb_migration.migrations._20230313164200_cache_remove_worker_version import (
    MigrationRemoveWorkerVersionFromCachedResponse,
)
from mongodb_migration.migrations._20230511100600_queue_remove_force import (
    MigrationRemoveForceFromJob,
)
from mongodb_migration.migrations._20230511100700_queue_delete_indexes_with_force import (
    MigrationQueueDeleteIndexesWithForce,
)
from mongodb_migration.migrations._20230511110700_queue_delete_skipped_jobs import (
    MigrationDeleteSkippedJobs,
)
from mongodb_migration.migrations._20230516101500_queue_job_add_revision import (
    MigrationQueueAddRevisionToJob,
)
from mongodb_migration.migrations._20230516101600_queue_delete_index_without_revision import (
    MigrationQueueDeleteIndexWithoutRevision,
)
from mongodb_migration.migrations._20230622131500_lock_add_owner import (
    MigrationAddOwnerToQueueLock,
)
from mongodb_migration.migrations._20230703110100_cache_add_partial_field_in_config_parquet_and_info import (
    MigrationAddPartialToCacheResponse,
)
from mongodb_migration.migrations._20230705160600_queue_job_add_difficulty import (
    MigrationQueueAddDifficultyToJob,
)
from mongodb_migration.migrations._20230926095900_cache_add_has_fts_field_in_split_duckdb_index import (
    MigrationAddHasFTSToSplitDuckdbIndexCacheResponse,
)
from mongodb_migration.renaming_migrations import (
    CacheRenamingMigration,
    QueueRenamingMigration,
)


# TODO: add a way to automatically collect migrations from the migrations/ folder
class MigrationsCollector:
    def get_migrations(self) -> list[Migration]:
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
            CacheRenamingMigration(
                cache_kind="/split-names",
                new_cache_kind="/split-names-from-streaming",
                version="20230216112500",
            ),
            QueueRenamingMigration(
                job_type="/split-names",
                new_job_type="/split-names-from-streaming",
                version="20230216141000",
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
            CacheRenamingMigration(
                cache_kind="/first-rows",
                new_cache_kind="split-first-rows-from-streaming",
                version="20230320163700",
            ),
            QueueRenamingMigration(
                job_type="/first-rows",
                new_job_type="split-first-rows-from-streaming",
                version="20230320165700",
            ),
            CacheRenamingMigration(
                cache_kind="/dataset-info",
                new_cache_kind="dataset-info",
                version="20230323155000",
            ),
            QueueRenamingMigration(
                job_type="/dataset-info",
                new_job_type="dataset-info",
                version="20230323160000",
            ),
            QueueDeletionMigration(
                job_type="/splits",
                version="20230407091400",
            ),
            CacheDeletionMigration(
                cache_kind="/splits",
                version="20230407091500",
            ),
            QueueDeletionMigration(
                job_type="/parquet-and-dataset-info",
                version="20230424173000",
            ),
            CacheDeletionMigration(
                cache_kind="/parquet-and-dataset-info",
                version="20230424174000",
            ),
            MetricsDeletionMigration(
                job_type="/parquet-and-dataset-info",
                cache_kind="/parquet-and-dataset-info",
                version="20230427121500",
            ),
            MigrationQueueDeleteTTLIndex(
                version="20230428145000",
                description="delete the TTL index on the 'finished_at' field in the queue database",
                field_name="finished_at",
            ),
            CacheDeletionMigration(
                cache_kind="dataset-split-names-from-streaming",
                version="20230428175100",
            ),
            QueueDeletionMigration(
                job_type="dataset-split-names-from-streaming",
                version="20230428181800",
            ),
            MetricsDeletionMigration(
                job_type="dataset-split-names-from-streaming",
                cache_kind="dataset-split-names-from-streaming",
                version="20230428193100",
            ),
            CacheDeletionMigration(
                cache_kind="dataset-split-names-from-dataset-info",
                version="20230504185100",
            ),
            QueueDeletionMigration(
                job_type="dataset-split-names-from-dataset-info",
                version="20230504192200",
            ),
            MetricsDeletionMigration(
                job_type="dataset-split-names-from-dataset-info",
                cache_kind="dataset-split-names-from-dataset-info",
                version="20230504194600",
            ),
            MigrationRemoveForceFromJob(version="20230511100600", description="remove 'force' field from queue"),
            MigrationQueueDeleteIndexesWithForce(
                version="20230511100700", description="remove indexes with field 'force'"
            ),
            MigrationDeleteSkippedJobs(version="20230511110700", description="delete jobs with skipped status"),
            MigrationQueueAddRevisionToJob(
                version="20230516101500", description="add 'revision' field to jobs in queue database"
            ),
            MigrationQueueDeleteIndexWithoutRevision(
                version="20230516101600", description="remove index without revision"
            ),
            CacheRenamingMigration(
                cache_kind="/split-names-from-streaming",
                new_cache_kind="config-split-names-from-streaming",
                version="20230516164500",
            ),
            QueueRenamingMigration(
                job_type="/split-names-from-streaming",
                new_job_type="config-split-names-from-streaming",
                version="20230516164700",
            ),
            MetricsDeletionMigration(
                job_type="/split-names-from-streaming",
                cache_kind="/split-names-from-streaming",
                version="20230522094400",
            ),
            MigrationQueueDeleteTTLIndex(
                version="20230523171700",
                description=(
                    "delete the TTL index on the 'finished_at' field in the queue database to update its TTL value"
                ),
                field_name="finished_at",
            ),
            CacheRenamingMigration(
                cache_kind="/split-names-from-dataset-info",
                new_cache_kind="config-split-names-from-info",
                version="20230524095900",
            ),
            QueueRenamingMigration(
                job_type="/split-names-from-dataset-info",
                new_job_type="config-split-names-from-info",
                version="20230524095901",
            ),
            MetricsDeletionMigration(
                job_type="/split-names-from-dataset-info",
                cache_kind="/split-names-from-dataset-info",
                version="20230524095902",
            ),
            CacheRenamingMigration(
                cache_kind="/config-names", new_cache_kind="dataset-config-names", version="20230524192200"
            ),
            QueueRenamingMigration(
                job_type="/config-names",
                new_job_type="dataset-config-names",
                version="20230524192300",
            ),
            MetricsDeletionMigration(job_type="/config-names", cache_kind="/config-names", version="20230524192400"),
            MigrationQueueDeleteTTLIndex(
                version="20230607154800",
                description=(
                    "delete the TTL index on the 'finished_at' field in the queue database to update its TTL condition"
                ),
                field_name="finished_at",
            ),
            MigrationQueueDeleteTTLIndex(
                version="202306201100",
                description=(
                    "delete the TTL index on the 'finished_at' field in the queue database to update its TTL condition"
                ),
                field_name="finished_at",
            ),
            MigrationAddOwnerToQueueLock(
                version="20230622131800", description="add 'owner' field copying the job_id value"
            ),
            MigrationAddPartialToCacheResponse(
                version="20230703110100", description="add 'partial' field to config-parquet-and-info"
            ),
            MigrationQueueAddDifficultyToJob(version="20230705160600", description="add 'difficulty' field to jobs"),
            MigrationDropCollection(
                version="20230811063600",
                description="drop cache metrics collection",
                alias=METRICS_MONGOENGINE_ALIAS,
                collection_name=CACHE_METRICS_COLLECTION,
            ),
            MigrationDropCollection(
                version="20230814121400",
                description="drop queue metrics collection",
                alias=METRICS_MONGOENGINE_ALIAS,
                collection_name=QUEUE_METRICS_COLLECTION,
            ),
            MigrationAddHasFTSToSplitDuckdbIndexCacheResponse(
                version="20230926095900",
                description="add 'has_fts' field for 'split-duckdb-index' cache records",
            ),
        ]
