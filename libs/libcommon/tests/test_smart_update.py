# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.
import pytest
from requests.exceptions import RequestException

from libcommon.orchestrator import (
    SmartUpdateImpossibleBecauseCacheIsEmpty,
    SmartUpdateImpossibleBecauseOfUpdatedFiles,
    SmartUpdateImpossibleBecauseOfUpdatedYAMLField,
)
from libcommon.resources import CacheMongoResource, QueueMongoResource

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
    assert_smart_dataset_update_plan(plan, tasks=[])


def test_empty_commit() -> None:
    # Empty commit: update the revision of the cache entries
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(EMPTY_DIFF):
        plan = get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)
        assert_smart_dataset_update_plan(
            plan,
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
            files_impacted_by_commit=["README.md"],
            updated_yaml_fields_in_dataset_card=[],
            tasks=["UpdateRevisionOfDatasetCacheEntriesTask,1"],
        )


def test_add_data() -> None:
    # Add data.txt commit: raise
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(ADD_DATA_DIFF):
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
    with put_diff(ADD_SECOND_CONFIG_DIFF), put_readme(ONE_CONFIG_README, revision=OTHER_REVISION_NAME), put_readme(
        TWO_CONFIGS_README
    ):
        with pytest.raises(SmartUpdateImpossibleBecauseOfUpdatedYAMLField):
            get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)


def test_add_tag_commit() -> None:
    # Add tag: update the revision of the cache entries
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)
    with put_diff(ADD_TAG_DIFF), put_readme(ONE_CONFIG_README, revision=OTHER_REVISION_NAME), put_readme(
        ONE_CONFIG_AND_ONE_TAG_README
    ):
        plan = get_smart_dataset_update_plan(processing_graph=PROCESSING_GRAPH_TWO_STEPS)
        assert_smart_dataset_update_plan(
            plan,
            files_impacted_by_commit=["README.md"],
            updated_yaml_fields_in_dataset_card=["tags"],
            tasks=["UpdateRevisionOfDatasetCacheEntriesTask,1"],
        )
