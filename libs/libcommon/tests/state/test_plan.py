# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List, Set

import pytest

from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource

from .utils import (
    DATASET_NAME,
    assert_dataset_state,
    compute_all,
    get_dataset_state,
    process_next_job,
    put_cache,
)

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

ARTIFACT_DA = f"{STEP_DA},{DATASET_NAME}"
ARTIFACT_DB = f"{STEP_DB},{DATASET_NAME}"
ARTIFACT_DC = f"{STEP_DC},{DATASET_NAME}"
ARTIFACT_DD = f"{STEP_DD},{DATASET_NAME}"
ARTIFACT_DE = f"{STEP_DE},{DATASET_NAME}"
ARTIFACT_DF = f"{STEP_DF},{DATASET_NAME}"
ARTIFACT_DG = f"{STEP_DG},{DATASET_NAME}"
ARTIFACT_DH = f"{STEP_DH},{DATASET_NAME}"
ARTIFACT_DI = f"{STEP_DI},{DATASET_NAME}"

STEP_CA = "config-a"
STEP_CB = "config-b"

ARTIFACT_CA_1 = f"{STEP_CA},{DATASET_NAME},{CONFIG_NAME_1}"
ARTIFACT_CA_2 = f"{STEP_CA},{DATASET_NAME},{CONFIG_NAME_2}"
ARTIFACT_CB_1 = f"{STEP_CB},{DATASET_NAME},{CONFIG_NAME_1}"
ARTIFACT_CB_2 = f"{STEP_CB},{DATASET_NAME},{CONFIG_NAME_2}"

STEP_SA = "split-a"

ARTIFACT_SA_1_1 = f"{STEP_SA},{DATASET_NAME},{CONFIG_NAME_1},{SPLIT_NAME_1}"
ARTIFACT_SA_1_2 = f"{STEP_SA},{DATASET_NAME},{CONFIG_NAME_1},{SPLIT_NAME_2}"
ARTIFACT_SA_2_1 = f"{STEP_SA},{DATASET_NAME},{CONFIG_NAME_2},{SPLIT_NAME_1}"
ARTIFACT_SA_2_2 = f"{STEP_SA},{DATASET_NAME},{CONFIG_NAME_2},{SPLIT_NAME_2}"


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
    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
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
        tasks=[f"CreateJob,{name}" for name in cache_is_empty],
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
    put_cache(ARTIFACT_DA)

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
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
        tasks=[f"CreateJob,{name}" for name in cache_is_empty],
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
    put_cache(ARTIFACT_DA)
    put_cache(ARTIFACT_CA_1)

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
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
        tasks=[f"CreateJob,{name}" for name in cache_is_empty],
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
    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
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
        tasks=[f"CreateJob,{name}" for name in new_1],
    )

    dataset_state.backfill()

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
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

    process_next_job(ARTIFACT_DA)

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
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
        tasks=[f"CreateJob,{name}" for name in new_2],
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

        dataset_state = get_dataset_state(processing_graph=processing_graph)
        assert_dataset_state(
            dataset_state=dataset_state,
            cache_status={
                "cache_has_different_git_revision": [],
                "cache_is_outdated_by_parent": is_outdated_by_parent,
                "cache_is_empty": is_empty,
                "cache_is_error_to_retry": [],
                "cache_is_job_runner_obsolete": [],
                "up_to_date": up_to_date,
            },
            queue_status={"in_process": []},
            tasks=[f"CreateJob,{name}" for name in in_process],
        )

        dataset_state.backfill()

        dataset_state = get_dataset_state(processing_graph=processing_graph)
        assert_dataset_state(
            dataset_state=dataset_state,
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

        for artifact in in_process:
            # note that they are updated in topological order (manually, in parametrize)
            process_next_job(artifact)


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

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
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

    put_cache(ARTIFACT_DA, error_code=error_code)
    # in the case of PROCESSING_GRAPH_FAN_IN_OUT: the config names do not exist anymore:
    # the cache entries (also the jobs, if any - not here) should be deleted.
    # they are still here, and haunting the database
    # TODO: Not supported yet

    dataset_state = get_dataset_state(processing_graph=processing_graph, error_codes_to_retry=error_codes_to_retry)
    assert_dataset_state(
        dataset_state=dataset_state,
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
        tasks=sorted([f"CreateJob,{ARTIFACT_DA}"] + [f"CreateJob,{name}" for name in is_outdated_by_parent]),
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

    put_cache(ARTIFACT_DA)

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": is_outdated_by_parent,
            "cache_is_empty": [],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=sorted([f"CreateJob,{name}" for name in is_outdated_by_parent]),
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

    put_cache(ARTIFACT_DA, use_old_job_runner_version=True)

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": is_outdated_by_parent,
            "cache_is_empty": [],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [ARTIFACT_DA],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=sorted([f"CreateJob,{ARTIFACT_DA}"] + [f"CreateJob,{name}" for name in is_outdated_by_parent]),
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

    put_cache(ARTIFACT_DA, use_other_git_revision=True)

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
        cache_status={
            "cache_has_different_git_revision": [ARTIFACT_DA],
            "cache_is_outdated_by_parent": is_outdated_by_parent,
            "cache_is_empty": [],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=sorted([f"CreateJob,{ARTIFACT_DA}"] + [f"CreateJob,{name}" for name in is_outdated_by_parent]),
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

    put_cache(ARTIFACT_SA_1_1)

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": is_outdated_by_parent,
            "cache_is_empty": [],
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=sorted([f"CreateJob,{name}" for name in is_outdated_by_parent]),
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
        put_cache(artifact=artifact)

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
        cache_status={
            "cache_has_different_git_revision": [],
            "cache_is_outdated_by_parent": [],
            "cache_is_empty": is_empty,
            "cache_is_error_to_retry": [],
            "cache_is_job_runner_obsolete": [],
            "up_to_date": up_to_date,
        },
        queue_status={"in_process": []},
        tasks=sorted([f"CreateJob,{name}" for name in is_empty]),
    )

    compute_all(processing_graph=processing_graph)

    dataset_state = get_dataset_state(processing_graph=processing_graph)
    assert_dataset_state(
        dataset_state=dataset_state,
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
