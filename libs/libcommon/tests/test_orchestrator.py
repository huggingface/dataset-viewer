# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from datetime import datetime
from typing import List, Optional, Set, Tuple

import pytest

from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.utils import Priority, Status

from .utils import (
    DATASET_NAME,
    REVISION_NAME,
    assert_dataset_backfill_plan,
    compute_all,
    get_dataset_backfill_plan,
    process_all_jobs,
    process_next_job,
    put_cache,
)

OTHER_REVISION_NAME = f"other_{REVISION_NAME}"

CONFIG_NAME_1 = "config1"
CONFIG_NAME_2 = "config2"
CONFIG_NAMES = [CONFIG_NAME_1, CONFIG_NAME_2]
CONFIG_NAMES_CONTENT = {"config_names": [{"config": config_name} for config_name in CONFIG_NAMES]}

SPLIT_NAME_1 = "split1"
SPLIT_NAME_2 = "split2"
SPLIT_NAMES = [SPLIT_NAME_1, SPLIT_NAME_2]
SPLIT_NAMES_CONTENT = {
    "splits": [{"dataset": DATASET_NAME, "config": CONFIG_NAME_1, "split": split_name} for split_name in SPLIT_NAMES]
}


STEP_DA = "dataset-a"
STEP_DB = "dataset-b"
STEP_DC = "dataset-c"
STEP_DD = "dataset-d"
STEP_DE = "dataset-e"
STEP_DF = "dataset-f"
STEP_DG = "dataset-g"
STEP_DH = "dataset-h"
STEP_DI = "dataset-i"

ARTIFACT_DA = f"{STEP_DA},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DA_OTHER_REVISION = f"{STEP_DA},{DATASET_NAME},{OTHER_REVISION_NAME}"
ARTIFACT_DB = f"{STEP_DB},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DC = f"{STEP_DC},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DD = f"{STEP_DD},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DE = f"{STEP_DE},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DF = f"{STEP_DF},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DG = f"{STEP_DG},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DH = f"{STEP_DH},{DATASET_NAME},{REVISION_NAME}"
ARTIFACT_DI = f"{STEP_DI},{DATASET_NAME},{REVISION_NAME}"

STEP_CA = "config-a"
STEP_CB = "config-b"

ARTIFACT_CA_1 = f"{STEP_CA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_1}"
ARTIFACT_CA_2 = f"{STEP_CA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_2}"
ARTIFACT_CB_1 = f"{STEP_CB},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_1}"
ARTIFACT_CB_2 = f"{STEP_CB},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_2}"

STEP_SA = "split-a"

ARTIFACT_SA_1_1 = f"{STEP_SA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_1},{SPLIT_NAME_1}"
ARTIFACT_SA_1_2 = f"{STEP_SA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_1},{SPLIT_NAME_2}"
ARTIFACT_SA_2_1 = f"{STEP_SA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_2},{SPLIT_NAME_1}"
ARTIFACT_SA_2_2 = f"{STEP_SA},{DATASET_NAME},{REVISION_NAME},{CONFIG_NAME_2},{SPLIT_NAME_2}"


# Graph to test only one step
#
#    +-------+
#    | DA    |
#    +-------+
#
PROCESSING_GRAPH_ONE_STEP = ProcessingGraph(
    processing_graph_specification={
        STEP_DA: {"input_type": "dataset"},
    }
)

# Graph to test siblings, children, grand-children, multiple parents
#
#    +-------+ +-------+
#    | DA    | | DB    |
#    +-------+ +-------+
#      |        |
#      |   +----+
#      |   |    |
#    +-------+  |
#    | DC    |  |
#    +-------+  |
#      |        |
#      |   +----+
#      |   |
#    +-------+
#    | DD    |
#    +-------+
#
PROCESSING_GRAPH_GENEALOGY = ProcessingGraph(
    processing_graph_specification={
        STEP_DA: {"input_type": "dataset", "provides_dataset_config_names": True},
        STEP_DB: {"input_type": "dataset"},  # sibling
        STEP_DC: {"input_type": "dataset", "triggered_by": [STEP_DA, STEP_DB]},  # child
        STEP_DD: {"input_type": "dataset", "triggered_by": [STEP_DB, STEP_DC]},  # grandchild
    }
)

# Graph to test fan-in, fan-out
#
#    +-------+
#    | DA    |
#    +-------+
#      |
#      ⩚
#    +-------+
#    | CA    |
#    +-------+
#      |   ⩛
#      |   +-----+
#      ⩚         |
#    +-------+ +-------+
#    | SA    | | DE    |
#    +-------+ +-------+
#      ⩛   ⩛
#      |   +-----+
#      |         |
#    +-------+ +-------+
#    | CB    | | DF    |
#    +-------+ +-------+
#
PROCESSING_GRAPH_FAN_IN_OUT = ProcessingGraph(
    processing_graph_specification={
        STEP_DA: {"input_type": "dataset", "provides_dataset_config_names": True},
        STEP_CA: {
            "input_type": "config",
            "triggered_by": STEP_DA,
            "provides_config_split_names": True,
        },  # fan-out (D->C)
        STEP_SA: {"input_type": "split", "triggered_by": STEP_CA},  # fan-out (C -> S)
        # is fan-out (D -> S) possible? (we need the list of split names anyway)
        STEP_DE: {"input_type": "dataset", "triggered_by": STEP_CA},  # fan-in (C -> D)
        STEP_CB: {"input_type": "config", "triggered_by": STEP_SA},  # fan-in (S -> C)
        STEP_DF: {"input_type": "dataset", "triggered_by": STEP_SA},  # fan-in (S -> D)
    }
)

# Graph to test parallel steps (ie. two steps that compute the same thing, and abort if the other already exists)
#
#    +-------+
#    | DA    |
#    +-------+
#      |
#      +---------+
#      |         |
#    +-------+ +-------+
#    | DG    | | DH    |
#    +-------+ +-------+
#      |         |
#      +---------+
#      |
#    +-------+
#    | DI    |
#    +-------+
#
PROCESSING_GRAPH_PARALLEL = ProcessingGraph(
    processing_graph_specification={
        STEP_DA: {"input_type": "dataset", "provides_dataset_config_names": True},
        STEP_DG: {"input_type": "dataset", "triggered_by": STEP_DA},
        STEP_DH: {"input_type": "dataset", "triggered_by": STEP_DA},
        STEP_DI: {"input_type": "dataset", "triggered_by": [STEP_DG, STEP_DH]},
    }
)


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


@pytest.mark.parametrize(
    "processing_graph,cache_is_empty",
    [
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DA, ARTIFACT_DB, ARTIFACT_DC, ARTIFACT_DD]),
        (PROCESSING_GRAPH_FAN_IN_OUT, [ARTIFACT_DA, ARTIFACT_DE, ARTIFACT_DF]),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DA, ARTIFACT_DG, ARTIFACT_DH, ARTIFACT_DI]),
    ],
)
def test_initial_state(
    processing_graph: ProcessingGraph,
    cache_is_empty: List[str],
) -> None:
    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=[],
        split_names_in_first_config=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": cache_is_empty,
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        queue_status={"in_process": []},
        tasks=[f"CreateJobs,{len(cache_is_empty)}"],
    )


@pytest.mark.parametrize(
    "processing_graph,cache_is_empty",
    [
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DB, ARTIFACT_DC, ARTIFACT_DD]),
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [ARTIFACT_CA_1, ARTIFACT_CA_2, ARTIFACT_CB_1, ARTIFACT_CB_2, ARTIFACT_DE, ARTIFACT_DF],
        ),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DG, ARTIFACT_DH, ARTIFACT_DI]),
    ],
)
def test_da_is_computed(
    processing_graph: ProcessingGraph,
    cache_is_empty: List[str],
) -> None:
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=REVISION_NAME)

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=CONFIG_NAMES,
        split_names_in_first_config=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": cache_is_empty,
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [ARTIFACT_DA],
        },
        queue_status={"in_process": []},
        tasks=[f"CreateJobs,{len(cache_is_empty)}"],
    )


@pytest.mark.parametrize(
    "processing_graph,cache_is_empty",
    [
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [ARTIFACT_CA_2, ARTIFACT_CB_1, ARTIFACT_CB_2, ARTIFACT_DE, ARTIFACT_DF, ARTIFACT_SA_1_1, ARTIFACT_SA_1_2],
        ),
    ],
)
def test_ca_1_is_computed(
    processing_graph: ProcessingGraph,
    cache_is_empty: List[str],
) -> None:
    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=REVISION_NAME)
    put_cache(step=STEP_CA, dataset=DATASET_NAME, revision=REVISION_NAME, config=CONFIG_NAME_1)

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=CONFIG_NAMES,
        split_names_in_first_config=SPLIT_NAMES,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": cache_is_empty,
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [ARTIFACT_CA_1, ARTIFACT_DA],
        },
        queue_status={"in_process": []},
        tasks=[f"CreateJobs,{len(cache_is_empty)}"],
    )


@pytest.mark.parametrize(
    "processing_graph,new_1,in_process_2,new_2",
    [
        (
            PROCESSING_GRAPH_GENEALOGY,
            [ARTIFACT_DA, ARTIFACT_DB, ARTIFACT_DC, ARTIFACT_DD],
            [ARTIFACT_DB, ARTIFACT_DC, ARTIFACT_DD],
            [],
        ),
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [ARTIFACT_DA, ARTIFACT_DE, ARTIFACT_DF],
            [ARTIFACT_DE, ARTIFACT_DF],
            [ARTIFACT_CA_1, ARTIFACT_CA_2, ARTIFACT_CB_1, ARTIFACT_CB_2],
        ),
        (
            PROCESSING_GRAPH_PARALLEL,
            [ARTIFACT_DA, ARTIFACT_DG, ARTIFACT_DH, ARTIFACT_DI],
            [ARTIFACT_DG, ARTIFACT_DH, ARTIFACT_DI],
            [],
        ),
    ],
)
def test_plan_one_job_creation_and_termination(
    processing_graph: ProcessingGraph, new_1: List[str], in_process_2: List[str], new_2: List[str]
) -> None:
    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=[],
        split_names_in_first_config=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": new_1,
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        queue_status={"in_process": []},
        tasks=[f"CreateJobs,{len(new_1)}"],
    )

    dataset_backfill_plan.run()

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=[],
        split_names_in_first_config=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": new_1,
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        queue_status={"in_process": new_1},
        tasks=[],
    )

    process_next_job()

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=CONFIG_NAMES,
        split_names_in_first_config=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": sorted(in_process_2 + new_2),
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [ARTIFACT_DA],
        },
        queue_status={"in_process": in_process_2},
        tasks=[f"CreateJobs,{len(new_2)}"] if new_2 else [],
    )


@pytest.mark.parametrize(
    "processing_graph,to_backfill",
    [
        (
            PROCESSING_GRAPH_GENEALOGY,
            [{ARTIFACT_DA, ARTIFACT_DB, ARTIFACT_DC, ARTIFACT_DD}, set()],
        ),
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [
                {ARTIFACT_DA, ARTIFACT_DE, ARTIFACT_DF},
                {ARTIFACT_CA_1, ARTIFACT_CA_2, ARTIFACT_CB_1, ARTIFACT_CB_2},
                {ARTIFACT_SA_1_1, ARTIFACT_SA_1_2, ARTIFACT_SA_2_1, ARTIFACT_SA_2_2, ARTIFACT_DE},
                {ARTIFACT_CB_1, ARTIFACT_CB_2, ARTIFACT_DF},
                set(),
            ],
        ),
        (PROCESSING_GRAPH_PARALLEL, [{ARTIFACT_DA, ARTIFACT_DG, ARTIFACT_DH, ARTIFACT_DI}, set()]),
    ],
)
def test_plan_all_job_creation_and_termination(processing_graph: ProcessingGraph, to_backfill: List[Set[str]]) -> None:
    previous_artifacts: Set[str] = set()
    for artifacts_to_backfill in to_backfill:
        is_empty = sorted(artifacts_to_backfill - previous_artifacts)
        is_outdated_by_parent = sorted(artifacts_to_backfill.intersection(previous_artifacts))
        in_process = sorted(is_empty + is_outdated_by_parent)
        up_to_date = sorted(previous_artifacts - artifacts_to_backfill)
        previous_artifacts = artifacts_to_backfill.union(previous_artifacts)

        dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
        assert_dataset_backfill_plan(
            dataset_backfill_plan=dataset_backfill_plan,
            cache_status={
                "cache_has_different_git_revision": [],
                "cache_is_outdated_by_parent": is_outdated_by_parent,
                "cache_is_empty": is_empty,
                "cache_is_error_to_retry": [],
                "cache_is_job_runner_obsolete": [],
                "up_to_date": up_to_date,
            },
            queue_status={"in_process": []},
            tasks=[f"CreateJobs,{len(in_process)}"] if in_process else [],
        )

        dataset_backfill_plan.run()

        dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
        assert_dataset_backfill_plan(
            dataset_backfill_plan=dataset_backfill_plan,
            cache_status={
                "cache_has_different_git_revision": [],
                "cache_is_outdated_by_parent": is_outdated_by_parent,
                "cache_is_empty": is_empty,
                "cache_is_error_to_retry": [],
                "cache_is_job_runner_obsolete": [],
                "up_to_date": up_to_date,
            },
            queue_status={"in_process": in_process},
            tasks=[],
        )

        process_all_jobs()


@pytest.mark.parametrize(
    "processing_graph,up_to_date",
    [
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DA, ARTIFACT_DB, ARTIFACT_DC, ARTIFACT_DD]),
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [
                ARTIFACT_CA_1,
                ARTIFACT_CA_2,
                ARTIFACT_CB_1,
                ARTIFACT_CB_2,
                ARTIFACT_DA,
                ARTIFACT_DE,
                ARTIFACT_DF,
                ARTIFACT_SA_1_1,
                ARTIFACT_SA_1_2,
                ARTIFACT_SA_2_1,
                ARTIFACT_SA_2_2,
            ],
        ),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DA, ARTIFACT_DG, ARTIFACT_DH, ARTIFACT_DI]),
    ],
)
def test_plan_compute_all(processing_graph: ProcessingGraph, up_to_date: List[str]) -> None:
    compute_all(processing_graph=processing_graph)

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=[],
    )


@pytest.mark.parametrize(
    "processing_graph,up_to_date,is_outdated_by_parent",
    [
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DB, ARTIFACT_DD], [ARTIFACT_DC]),
        (PROCESSING_GRAPH_FAN_IN_OUT, [ARTIFACT_DE, ARTIFACT_DF], []),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DI], [ARTIFACT_DG, ARTIFACT_DH]),
    ],
)
def test_plan_retry_error_and_outdated_by_parent(
    processing_graph: ProcessingGraph, up_to_date: List[str], is_outdated_by_parent: List[str]
) -> None:
    error_code = "ERROR_CODE_TO_RETRY"
    error_codes_to_retry = [error_code]
    compute_all(processing_graph=processing_graph, error_codes_to_retry=error_codes_to_retry)

    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=REVISION_NAME, error_code=error_code)
    # in the case of PROCESSING_GRAPH_FAN_IN_OUT: the config names do not exist anymore:
    # the cache entries (also the jobs, if any - not here) should be deleted.
    # they are still here, and haunting the database
    # TODO: Not supported yet

    dataset_backfill_plan = get_dataset_backfill_plan(
        processing_graph=processing_graph, error_codes_to_retry=error_codes_to_retry
    )
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": is_outdated_by_parent,
            "cache_is_empty": [],
            "cache_is_error_to_retry": [ARTIFACT_DA],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=[f"CreateJobs,{len(is_outdated_by_parent) + 1}"],
    )


@pytest.mark.parametrize(
    "processing_graph,up_to_date,is_outdated_by_parent",
    [
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DA, ARTIFACT_DB, ARTIFACT_DD], [ARTIFACT_DC]),
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [
                ARTIFACT_CB_1,
                ARTIFACT_CB_2,
                ARTIFACT_DA,
                ARTIFACT_DE,
                ARTIFACT_DF,
                ARTIFACT_SA_1_1,
                ARTIFACT_SA_1_2,
                ARTIFACT_SA_2_1,
                ARTIFACT_SA_2_2,
            ],
            [ARTIFACT_CA_1, ARTIFACT_CA_2],
        ),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DA, ARTIFACT_DI], [ARTIFACT_DG, ARTIFACT_DH]),
    ],
)
def test_plan_outdated_by_parent(
    processing_graph: ProcessingGraph, up_to_date: List[str], is_outdated_by_parent: List[str]
) -> None:
    compute_all(processing_graph=processing_graph)

    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=REVISION_NAME)

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": is_outdated_by_parent,
            "cache_is_empty": [],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=[f"CreateJobs,{len(is_outdated_by_parent)}"],
    )


@pytest.mark.parametrize(
    "processing_graph,up_to_date,is_outdated_by_parent",
    [
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DB, ARTIFACT_DD], [ARTIFACT_DC]),
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [
                ARTIFACT_CB_1,
                ARTIFACT_CB_2,
                ARTIFACT_DE,
                ARTIFACT_DF,
                ARTIFACT_SA_1_1,
                ARTIFACT_SA_1_2,
                ARTIFACT_SA_2_1,
                ARTIFACT_SA_2_2,
            ],
            [ARTIFACT_CA_1, ARTIFACT_CA_2],
        ),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DI], [ARTIFACT_DG, ARTIFACT_DH]),
    ],
)
def test_plan_job_runner_version_and_outdated_by_parent(
    processing_graph: ProcessingGraph, up_to_date: List[str], is_outdated_by_parent: List[str]
) -> None:
    compute_all(processing_graph=processing_graph)

    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=REVISION_NAME, use_old_job_runner_version=True)

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": is_outdated_by_parent,
            "cache_is_empty": [],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [ARTIFACT_DA],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=[f"CreateJobs,{len(is_outdated_by_parent) + 1}"],
    )


@pytest.mark.parametrize(
    "processing_graph,up_to_date,is_outdated_by_parent",
    [
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DB, ARTIFACT_DD], [ARTIFACT_DC]),
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [
                ARTIFACT_CB_1,
                ARTIFACT_CB_2,
                ARTIFACT_DE,
                ARTIFACT_DF,
                ARTIFACT_SA_1_1,
                ARTIFACT_SA_1_2,
                ARTIFACT_SA_2_1,
                ARTIFACT_SA_2_2,
            ],
            [ARTIFACT_CA_1, ARTIFACT_CA_2],
        ),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DI], [ARTIFACT_DG, ARTIFACT_DH]),
    ],
)
def test_plan_git_revision_and_outdated_by_parent(
    processing_graph: ProcessingGraph, up_to_date: List[str], is_outdated_by_parent: List[str]
) -> None:
    compute_all(processing_graph=processing_graph)

    put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=OTHER_REVISION_NAME)

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        cache_status={
            "cache_has_different_git_revision": [ARTIFACT_DA],
            "cache_is_outdated_by_parent": is_outdated_by_parent,
            "cache_is_empty": [],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=[f"CreateJobs,{len(is_outdated_by_parent) + 1}"],
    )


@pytest.mark.parametrize(
    "processing_graph,up_to_date,is_outdated_by_parent",
    [
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [
                ARTIFACT_CA_1,
                ARTIFACT_CA_2,
                ARTIFACT_CB_2,
                ARTIFACT_DA,
                ARTIFACT_DE,
                ARTIFACT_SA_1_1,
                ARTIFACT_SA_1_2,
                ARTIFACT_SA_2_1,
                ARTIFACT_SA_2_2,
            ],
            [
                ARTIFACT_CB_1,
                ARTIFACT_DF,
            ],
        ),
    ],
)
def test_plan_fan_in_updated(
    processing_graph: ProcessingGraph, up_to_date: List[str], is_outdated_by_parent: List[str]
) -> None:
    compute_all(processing_graph=processing_graph)

    put_cache(step=STEP_SA, dataset=DATASET_NAME, revision=REVISION_NAME, config=CONFIG_NAME_1, split=SPLIT_NAME_1)

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": is_outdated_by_parent,
            "cache_is_empty": [],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=[f"CreateJobs,{len(is_outdated_by_parent)}"],
    )


@pytest.mark.parametrize(
    "processing_graph,initial,up_to_date,is_empty,unknown",
    [
        (
            PROCESSING_GRAPH_GENEALOGY,
            [ARTIFACT_DA, ARTIFACT_DD],
            [ARTIFACT_DA, ARTIFACT_DD],
            [ARTIFACT_DB, ARTIFACT_DC],
            [],
        ),
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [ARTIFACT_CA_1],
            [],
            [ARTIFACT_DA, ARTIFACT_DE, ARTIFACT_DF],
            [
                ARTIFACT_CA_1,
                ARTIFACT_CA_2,
                ARTIFACT_CB_1,
                ARTIFACT_CB_2,
                ARTIFACT_SA_1_1,
                ARTIFACT_SA_1_2,
                ARTIFACT_SA_2_1,
                ARTIFACT_SA_2_2,
            ],
        ),
        (
            PROCESSING_GRAPH_FAN_IN_OUT,
            [ARTIFACT_SA_1_1],
            [],
            [ARTIFACT_DA, ARTIFACT_DE, ARTIFACT_DF],
            [
                ARTIFACT_CA_1,
                ARTIFACT_CA_2,
                ARTIFACT_CB_1,
                ARTIFACT_CB_2,
                ARTIFACT_SA_1_1,
                ARTIFACT_SA_1_2,
                ARTIFACT_SA_2_1,
                ARTIFACT_SA_2_2,
            ],
        ),
        (
            PROCESSING_GRAPH_PARALLEL,
            [ARTIFACT_DA, ARTIFACT_DI],
            [ARTIFACT_DA, ARTIFACT_DI],
            [ARTIFACT_DG, ARTIFACT_DH],
            [],
        ),
    ],
)
def test_plan_incoherent_state(
    processing_graph: ProcessingGraph,
    initial: List[str],
    up_to_date: List[str],
    is_empty: List[str],
    unknown: List[str],
) -> None:
    for artifact in initial:
        if artifact == ARTIFACT_SA_1_1:
            put_cache(
                step=STEP_SA, dataset=DATASET_NAME, revision=REVISION_NAME, config=CONFIG_NAME_1, split=SPLIT_NAME_1
            )
        elif artifact == ARTIFACT_CA_1:
            put_cache(step=STEP_CA, dataset=DATASET_NAME, revision=REVISION_NAME, config=CONFIG_NAME_1)
        elif artifact == ARTIFACT_DA:
            put_cache(step=STEP_DA, dataset=DATASET_NAME, revision=REVISION_NAME)
        elif artifact == ARTIFACT_DD:
            put_cache(step=STEP_DD, dataset=DATASET_NAME, revision=REVISION_NAME)
        elif artifact == ARTIFACT_DI:
            put_cache(step=STEP_DI, dataset=DATASET_NAME, revision=REVISION_NAME)
        else:
            raise NotImplementedError()

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": is_empty,
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=[f"CreateJobs,{len(is_empty)}"],
    )

    compute_all(processing_graph=processing_graph)

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": sorted(up_to_date + is_empty + unknown),
        },
        queue_status={"in_process": []},
        tasks=[],
    )


JobSpec = Tuple[Priority, Status, Optional[datetime]]

OLD = datetime.strptime("20000101", "%Y%m%d")
NEW = datetime.strptime("20000102", "%Y%m%d")
LOW_WAITING_OLD = (Priority.LOW, Status.WAITING, OLD)
LOW_WAITING_NEW = (Priority.LOW, Status.WAITING, NEW)
LOW_STARTED_OLD = (Priority.LOW, Status.STARTED, OLD)
LOW_STARTED_NEW = (Priority.LOW, Status.STARTED, NEW)
NORMAL_WAITING_OLD = (Priority.NORMAL, Status.WAITING, OLD)
NORMAL_WAITING_NEW = (Priority.NORMAL, Status.WAITING, NEW)
NORMAL_STARTED_OLD = (Priority.NORMAL, Status.STARTED, OLD)
NORMAL_STARTED_NEW = (Priority.NORMAL, Status.STARTED, NEW)


@pytest.mark.parametrize(
    "existing_jobs,expected_create_job,expected_delete_jobs,expected_jobs_after_backfill",
    [
        ([], True, False, [(Priority.LOW, Status.WAITING, None)]),
        (
            [
                LOW_WAITING_OLD,
                LOW_WAITING_NEW,
                LOW_STARTED_OLD,
                LOW_STARTED_NEW,
                NORMAL_WAITING_OLD,
                NORMAL_WAITING_NEW,
                NORMAL_STARTED_OLD,
                NORMAL_STARTED_NEW,
            ],
            False,
            True,
            [NORMAL_STARTED_OLD],
        ),
        (
            [
                LOW_WAITING_OLD,
                LOW_WAITING_NEW,
                LOW_STARTED_OLD,
                LOW_STARTED_NEW,
                NORMAL_WAITING_OLD,
                NORMAL_WAITING_NEW,
                NORMAL_STARTED_NEW,
            ],
            False,
            True,
            [NORMAL_STARTED_NEW],
        ),
        (
            [
                LOW_WAITING_OLD,
                LOW_WAITING_NEW,
                LOW_STARTED_OLD,
                LOW_STARTED_NEW,
                NORMAL_WAITING_OLD,
                NORMAL_WAITING_NEW,
            ],
            False,
            True,
            [LOW_STARTED_OLD],
        ),
        (
            [LOW_WAITING_OLD, LOW_WAITING_NEW, LOW_STARTED_NEW, NORMAL_WAITING_OLD, NORMAL_WAITING_NEW],
            False,
            True,
            [LOW_STARTED_NEW],
        ),
        (
            [LOW_WAITING_OLD, LOW_WAITING_NEW, NORMAL_WAITING_OLD, NORMAL_WAITING_NEW],
            False,
            True,
            [NORMAL_WAITING_OLD],
        ),
        ([LOW_WAITING_OLD, LOW_WAITING_NEW, NORMAL_WAITING_NEW], False, True, [NORMAL_WAITING_NEW]),
        ([LOW_WAITING_OLD, LOW_WAITING_NEW], False, True, [LOW_WAITING_OLD]),
        ([LOW_WAITING_NEW], False, False, [LOW_WAITING_NEW]),
        ([LOW_WAITING_NEW] * 5, False, True, [LOW_WAITING_NEW]),
    ],
)
def test_delete_jobs(
    existing_jobs: List[JobSpec],
    expected_create_job: bool,
    expected_delete_jobs: bool,
    expected_jobs_after_backfill: List[JobSpec],
) -> None:
    processing_graph = PROCESSING_GRAPH_ONE_STEP

    queue = Queue()
    for job_spec in existing_jobs:
        (priority, status, created_at) = job_spec
        job = queue._add_job(job_type=STEP_DA, dataset="dataset", revision="revision", priority=priority)
        if created_at is not None:
            job.created_at = created_at
            job.save()
        if status is Status.STARTED:
            queue._start_job(job)

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph)
    expected_in_process = [ARTIFACT_DA] if existing_jobs else []
    if expected_create_job:
        if expected_delete_jobs:
            raise NotImplementedError()
        expected_tasks = ["CreateJobs,1"]
    elif expected_delete_jobs:
        expected_tasks = [f"DeleteJobs,{len(existing_jobs) - 1}"]
    else:
        expected_tasks = []

    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=[],
        split_names_in_first_config=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [ARTIFACT_DA],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        queue_status={"in_process": expected_in_process},
        tasks=expected_tasks,
    )

    dataset_backfill_plan.run()

    job_dicts = queue.get_dataset_pending_jobs_for_type(dataset=DATASET_NAME, job_type=STEP_DA)
    assert len(job_dicts) == len(expected_jobs_after_backfill)
    for job_dict, expected_job_spec in zip(job_dicts, expected_jobs_after_backfill):
        (priority, status, created_at) = expected_job_spec
        assert job_dict["priority"] == priority.value
        assert job_dict["status"] == status.value
        if created_at is not None:
            assert job_dict["created_at"] == created_at


def test_multiple_revisions() -> None:
    processing_graph = PROCESSING_GRAPH_ONE_STEP

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph, revision=REVISION_NAME)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=[],
        split_names_in_first_config=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [ARTIFACT_DA],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        queue_status={"in_process": []},
        tasks=["CreateJobs,1"],
    )

    # create the job for the first revision
    dataset_backfill_plan.run()

    # the job is in process, no other job is created for the same revision
    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph, revision=REVISION_NAME)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=[],
        split_names_in_first_config=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [ARTIFACT_DA],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        queue_status={"in_process": [ARTIFACT_DA]},
        tasks=[],
    )

    # create the job for the second revision: the first job is deleted
    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph, revision=OTHER_REVISION_NAME)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=[],
        split_names_in_first_config=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [ARTIFACT_DA_OTHER_REVISION],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        queue_status={"in_process": []},
        tasks=["DeleteJobs,1", "CreateJobs,1"],
    )
    dataset_backfill_plan.run()

    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph=processing_graph, revision=OTHER_REVISION_NAME)
    assert_dataset_backfill_plan(
        dataset_backfill_plan=dataset_backfill_plan,
        config_names=[],
        split_names_in_first_config=[],
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": [ARTIFACT_DA_OTHER_REVISION],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": [],
        },
        queue_status={"in_process": [ARTIFACT_DA_OTHER_REVISION]},
        tasks=[],
    )
    pending_jobs_df = Queue().get_pending_jobs_df(dataset=DATASET_NAME)
    assert len(pending_jobs_df) == 1
    assert not (pending_jobs_df["revision"] == REVISION_NAME).any()
    assert (pending_jobs_df["revision"] == OTHER_REVISION_NAME).all()
