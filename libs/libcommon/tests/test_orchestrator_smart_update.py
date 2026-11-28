# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
import time
from collections.abc import Iterator
from unittest.mock import patch

import httpx2 as httpx
import pytest

from libcommon.orchestrator import (
    MAX_DIFF_BYTES,
    SmartUpdateImpossibleBecauseCachedRevisionIsNotParentOfNewRevision,
    SmartUpdateImpossibleBecauseCacheIsEmpty,
    SmartUpdateImpossibleBecauseDiffIsTooBig,
    SmartUpdateImpossibleBecauseOfUpdatedFiles,
    SmartUpdateImpossibleBecauseOfUpdatedYAMLField,
    TasksStatistics,
    get_impacted_files_from_diff,
)
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import get_cache_entries_df
from libcommon.storage_client import StorageClient
from libcommon.utils import thread_map

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

RENAME_FILE_DIFF = """
diff --git a/train.csv b/test.csv
similarity index 100%
rename from train.csv
rename to test.csv
"""

RENAME_README_DIFF = """
diff --git a/README.md b/README_old.md
similarity index 100%
rename from README.md
rename to README_old.md
"""

NO_TRAILING_NEWLINE_DIFF = "--- a/README.md\n+++ b/.gitattributes"


def endless_diff(header: bytes, line: bytes = b"+" + b"x" * 1022 + b"\n") -> Iterator[bytes]:
    # a diff of a commit that changes a big text file stored in git: it never ends
    yield header
    while True:
        yield line


def endless_data_diff() -> Iterator[bytes]:
    return endless_diff(b"diff --git a/data.txt b/data.txt\n--- a/data.txt\n+++ b/data.txt\n@@ -1 +1 @@\n")


def endless_readme_diff() -> Iterator[bytes]:
    return endless_diff(b"diff --git a/README.md b/README.md\n--- a/README.md\n+++ b/README.md\n@@ -1 +1 @@\n")


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


@pytest.mark.parametrize("chunk_size", [1, 7, 1024])
@pytest.mark.parametrize(
    "diff,expected",
    [
        (EMPTY_DIFF, set()),
        (ADD_INITIAL_README_DIFF, {"README.md"}),
        (ADD_CONFIG_DIFF, {"README.md"}),
        (ADD_DATA_DIFF, {"README.md", "data.csv"}),
        (RENAME_FILE_DIFF, {"train.csv"}),
        (RENAME_README_DIFF, {"README.md", "README_old.md"}),
        (NO_TRAILING_NEWLINE_DIFF, {"README.md", ".gitattributes"}),
    ],
)
def test_get_impacted_files_from_diff(diff: str, expected: set[str], chunk_size: int) -> None:
    encoded_diff = diff.encode("utf-8")
    chunks = (encoded_diff[i : i + chunk_size] for i in range(0, len(encoded_diff), chunk_size))
    assert get_impacted_files_from_diff(chunks) == expected


def test_get_impacted_files_from_diff_stops_at_the_first_other_file() -> None:
    assert get_impacted_files_from_diff(endless_data_diff()) == {"data.txt"}


@pytest.mark.parametrize("line", [b"+" + b"x" * 1022 + b"\n", b"x" * 1024])
def test_get_impacted_files_from_diff_too_big(line: bytes) -> None:
    num_bytes = 0

    def counted(chunks: Iterator[bytes]) -> Iterator[bytes]:
        nonlocal num_bytes
        for chunk in chunks:
            num_bytes += len(chunk)
            yield chunk

    header = b"--- a/README.md\n+++ b/README.md\n"
    with pytest.raises(SmartUpdateImpossibleBecauseDiffIsTooBig):
        get_impacted_files_from_diff(counted(endless_diff(header, line)))
    assert num_bytes <= MAX_DIFF_BYTES + len(line)


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
    with pytest.raises((httpx.ConnectError, httpx.HTTPStatusError)):  # if diff doesn't exist
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


def test_rename_file() -> None:
    # Rename train.csv -> test.csv commit: raise
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(RENAME_FILE_DIFF), put_readme(None):
        with pytest.raises(SmartUpdateImpossibleBecauseOfUpdatedFiles):
            get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_huge_data_diff() -> None:
    # Commit that changes a big text file stored in git: raise without reading the whole diff
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(endless_data_diff()), put_readme(None):
        with pytest.raises(SmartUpdateImpossibleBecauseOfUpdatedFiles):
            get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_huge_readme_diff() -> None:
    # Commit that only changes the README.md, but with a huge diff: raise, the dataset is updated normally
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(endless_readme_diff()), put_readme(None):
        with pytest.raises(SmartUpdateImpossibleBecauseDiffIsTooBig):
            get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_huge_diff_from_the_hub_is_streamed() -> None:
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    sent_bytes = 0

    def stream() -> Iterator[bytes]:
        nonlocal sent_bytes
        for chunk in endless_data_diff():
            sent_bytes += len(chunk)
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == f"/api/datasets/{DATASET_NAME}/compare/{REVISION_NAME}^..{REVISION_NAME}"
        return httpx.Response(200, content=stream())

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        with patch("libcommon.orchestrator.get_session", return_value=client), put_readme(None):
            with pytest.raises(SmartUpdateImpossibleBecauseOfUpdatedFiles):
                get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)
    assert sent_bytes < 1024 * 1024


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
    previous_key = storage_client.generate_object_path(
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

    updated_key = storage_client.generate_object_path(
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
