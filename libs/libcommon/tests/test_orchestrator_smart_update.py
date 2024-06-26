# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
import time

import pytest
from requests.exceptions import RequestException
from tqdm.contrib.concurrent import thread_map

from libcommon.orchestrator import (
    SmartUpdateImpossibleBecauseCachedRevisionIsNotParentOfNewRevision,
    SmartUpdateImpossibleBecauseCacheIsEmpty,
    SmartUpdateImpossibleBecauseOfUpdatedFiles,
    SmartUpdateImpossibleBecauseOfUpdatedYAMLField,
    TasksStatistics,
)
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import get_cache_entries_df
from libcommon.storage_client import StorageClient

from .utils import (
    DATASET_NAME,
    OTHER_REVISION_NAME,
    PROCESSING_GRAPH_TWO_STEPS,
    REVISION_NAME,
    STEP_DA,
    assert_smart_dataset_update_plan,
    get_smart_dataset_update_plan,
    put_cache,
    put_diff,
    put_readme,
)

EMPTY_DIFF = ""

ADD_INITIAL_README_DIFF = f"""
diff --git a/README.md b/README.md
new file mode 100644
index 0000000000000000000000000000000000000000..6ea47cd9c61754a4838b07f479b376cb431f2270
--- /dev/null
+++ b/README.md
@@ -0,0 +1,1 @@
+# Dataset Card for {DATASET_NAME}
"""

ADD_INITIAL_README_WITH_CONFIG_DIFF = f"""
diff --git a/README.md b/README.md
new file mode 100644
index 0000000000000000000000000000000000000000..6ea47cd9c61754a4838b07f479b376cb431f2270
--- /dev/null
+++ b/README.md
@@ -0,0 +1,7 @@
+---
+configs:
+- config_name: foo
+  data_files: foo.csv
+---
+
+# Dataset Card for {DATASET_NAME}
"""

ADD_CONFIG_DIFF = """
diff --git a/README.md b/README.md
new file mode 100644
index 6ea47cd9c61754a4838b07f479b376cb431f2270..6ea47cd9c61754a4838b07f479b376cb431f2271
--- a/README.md
+++ b/README.md
@@ -0,0 +1,6 @@
+---
+configs:
+- config_name: foo
+  data_files: foo.csv
+---
+
"""

ADD_SECOND_CONFIG_DIFF = """
diff --git a/README.md b/README.md
new file mode 100644
index 6ea47cd9c61754a4838b07f479b376cb431f2271..6ea47cd9c61754a4838b07f479b376cb431f2272
--- a/README.md
+++ b/README.md
@@ -0,0 +5,6 @@
+- config_name: bar
+  data_files: bar.csv
"""


ADD_TAG_DIFF = """
diff --git a/README.md b/README.md
new file mode 100644
index 6ea47cd9c61754a4838b07f479b376cb431f2271..6ea47cd9c61754a4838b07f479b376cb431f2273
--- a/README.md
+++ b/README.md
@@ -0,0 +5,6 @@
+tags:
+- test
"""

ADD_DATA_DIFF = """
diff --git a/data.csv b/data.csv
new file mode 100644
index 6ea47cd9c61754a4838b07f479b376cb431f2271..6ea47cd9c61754a4838b07f479b376cb431f2274
--- a/README.md
+++ b/data.csv
@@ -0,0 +1,3 @@
+id,text
+0,foo
+1,bar
"""

INITIAL_README = f"# Dataset Card for {DATASET_NAME}"
ONE_CONFIG_README = (
    f"---\nconfigs:\n- config_name: foo\n  data_files: foo.csv\n---\n\n# Dataset Card for {DATASET_NAME}"
)
TWO_CONFIGS_README = f"---\nconfigs:\n- config_name: foo\n  data_files: foo.csv\n- config_name: bar\n  data_files: bar.csv\n---\n\n# Dataset Card for {DATASET_NAME}"
ONE_CONFIG_AND_ONE_TAG_README = f"---\nconfigs:\n- config_name: foo\n  data_files: foo.csv\ntags:\n- test\n---\n\n# Dataset Card for {DATASET_NAME}"


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


def test_empty_cache() -> None:
    # No cache, raise
    with pytest.raises(SmartUpdateImpossibleBecauseCacheIsEmpty):
        get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_same_revision() -> None:
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=REVISION_NAME)
    # Same revision as cache, do nothing
    plan = get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)
    assert_smart_dataset_update_plan(plan, cached_revision=REVISION_NAME, tasks=[])


def test_cache_revision_is_not_parent_revision_commit() -> None:
    # If rewriting git history OR spamming commits OR out of order commits OR in case of concurrency issues: raise
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision="not_parent_revision")
    with pytest.raises(SmartUpdateImpossibleBecauseCachedRevisionIsNotParentOfNewRevision):
        get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_empty_commit() -> None:
    # Empty commit: update the revision of the cache entries
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(EMPTY_DIFF), put_readme(None):
        plan = get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)
        assert_smart_dataset_update_plan(
            plan,
            cached_revision=OTHER_REVISION_NAME,
            files_impacted_by_commit=[],
            tasks=["UpdateRevisionOfDatasetCacheEntriesTask,1"],
        )
    with pytest.raises(RequestException):  # if diff doesn't exist
        get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_add_initial_readme_commit() -> None:
    # Add README.md commit: update the revision of the cache entries
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(ADD_INITIAL_README_DIFF), put_readme(INITIAL_README):
        plan = get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)
        assert_smart_dataset_update_plan(
            plan,
            cached_revision=OTHER_REVISION_NAME,
            files_impacted_by_commit=["README.md"],
            updated_yaml_fields_in_dataset_card=[],
            tasks=["UpdateRevisionOfDatasetCacheEntriesTask,1"],
        )


def test_add_initial_readme_with_config_commit() -> None:
    # Add README.md commit: update the revision of the cache entries
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(ADD_INITIAL_README_WITH_CONFIG_DIFF), put_readme(ONE_CONFIG_README):
        with pytest.raises(SmartUpdateImpossibleBecauseOfUpdatedYAMLField):
            get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_add_data() -> None:
    # Add data.txt commit: raise
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(ADD_DATA_DIFF), put_readme(None):
        with pytest.raises(SmartUpdateImpossibleBecauseOfUpdatedFiles):
            get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_add_config_commit() -> None:
    # Add config commit: raise
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(ADD_CONFIG_DIFF), put_readme(ONE_CONFIG_README):
        with pytest.raises(SmartUpdateImpossibleBecauseOfUpdatedYAMLField):
            get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_add_second_config_commit() -> None:
    # Add a second config commit: raise
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with (
        put_diff(ADD_SECOND_CONFIG_DIFF),
        put_readme(ONE_CONFIG_README, revision=OTHER_REVISION_NAME),
        put_readme(TWO_CONFIGS_README),
    ):
        with pytest.raises(SmartUpdateImpossibleBecauseOfUpdatedYAMLField):
            get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_add_tag_commit() -> None:
    # Add tag: update the revision of the cache entries
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with (
        put_diff(ADD_TAG_DIFF),
        put_readme(ONE_CONFIG_README, revision=OTHER_REVISION_NAME),
        put_readme(ONE_CONFIG_AND_ONE_TAG_README),
    ):
        plan = get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)
        assert_smart_dataset_update_plan(
            plan,
            cached_revision=OTHER_REVISION_NAME,
            files_impacted_by_commit=["README.md"],
            updated_yaml_fields_in_dataset_card=["tags"],
            tasks=["UpdateRevisionOfDatasetCacheEntriesTask,1"],
        )


def test_run() -> None:
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(EMPTY_DIFF), put_readme(None):
        tasks_stats = get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS).run()
    assert tasks_stats.num_created_jobs == 0
    assert tasks_stats.num_updated_cache_entries == 1
    assert tasks_stats.num_updated_storage_directories == 0
    assert tasks_stats.num_deleted_cache_entries == 0
    assert tasks_stats.num_deleted_storage_directories == 0
    assert tasks_stats.num_deleted_waiting_jobs == 0

    cache_entries_df = get_cache_entries_df(DATASET_NAME, cache_kinds=[STEP_DA])
    assert len(cache_entries_df) == 1
    cache_entry = cache_entries_df.to_dict(orient="records")[0]
    assert cache_entry["dataset_git_revision"] == REVISION_NAME


def test_run_with_storage_clients(storage_client: StorageClient) -> None:
    filename = "object.asset"
    previous_key = storage_client.generate_object_key(
        dataset=DATASET_NAME,
        revision=OTHER_REVISION_NAME,
        config="default",
        split="train",
        row_idx=0,
        column="image",
        filename=filename,
    )
    storage_client._fs.touch(storage_client.get_full_path(previous_key))
    assert storage_client.exists(previous_key)
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(EMPTY_DIFF), put_readme(None):
        tasks_stats = get_smart_dataset_update_plan(
            processing_graph=PROCESSING_GRAPH_TWO_STEPS, storage_clients=[storage_client]
        ).run()
    assert tasks_stats.num_created_jobs == 0
    assert tasks_stats.num_updated_cache_entries == 1
    assert tasks_stats.num_updated_storage_directories == 1
    assert tasks_stats.num_deleted_cache_entries == 0
    assert tasks_stats.num_deleted_storage_directories == 0
    assert tasks_stats.num_deleted_waiting_jobs == 0

    cache_entries_df = get_cache_entries_df(DATASET_NAME, cache_kinds=[STEP_DA])
    assert len(cache_entries_df) == 1
    cache_entry = cache_entries_df.to_dict(orient="records")[0]
    assert cache_entry["dataset_git_revision"] == REVISION_NAME

    updated_key = storage_client.generate_object_key(
        dataset=DATASET_NAME,
        revision=REVISION_NAME,
        config="default",
        split="train",
        row_idx=0,
        column="image",
        filename=filename,
    )
    assert storage_client.exists(updated_key)


@pytest.mark.parametrize("out_of_order", [False, True])
def test_run_two_commits(out_of_order: bool) -> None:
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision="initial_revision")
    with (
        put_diff(ADD_TAG_DIFF, revision=REVISION_NAME),
        put_diff(ADD_TAG_DIFF, revision=OTHER_REVISION_NAME),
        put_readme(None, revision=REVISION_NAME),
        put_readme(None, revision=OTHER_REVISION_NAME),
    ):

        def run_plan(revisions: tuple[str, str]) -> TasksStatistics:
            old_revision, revision = revisions
            if (revision == REVISION_NAME) ^ out_of_order:
                time.sleep(0.5)
            return get_smart_dataset_update_plan(
                processing_graph=PROCESSING_GRAPH_TWO_STEPS, revision=revision, old_revision=old_revision
            ).run()

        stats: list[TasksStatistics] = thread_map(
            run_plan, [("initial_revision", OTHER_REVISION_NAME), (OTHER_REVISION_NAME, REVISION_NAME)]
        )
    assert stats[0].num_updated_cache_entries == 1
    assert stats[1].num_updated_cache_entries == 1

    cache_entries_df = get_cache_entries_df(DATASET_NAME, cache_kinds=[STEP_DA])
    assert len(cache_entries_df) == 1
    cache_entry = cache_entries_df.to_dict(orient="records")[0]
    assert cache_entry["dataset_git_revision"] == REVISION_NAME
