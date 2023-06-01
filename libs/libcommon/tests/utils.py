# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Dict, List, Optional

from libcommon.orchestrator import DatasetBackfillPlan
from libcommon.processing_graph import Artifact, ProcessingGraph
from libcommon.queue import Queue
from libcommon.simple_cache import upsert_response
from libcommon.utils import JobInfo, Priority

DATASET_NAME = "dataset"

REVISION_NAME = "revision"

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


CACHE_KIND = "cache_kind"
CONTENT_ERROR = {"error": "error"}
JOB_TYPE = "job_type"

STEP_DATASET_A = "dataset-a"
STEP_CONFIG_B = "config-b"
STEP_SPLIT_C = "split-c"
PROCESSING_GRAPH = ProcessingGraph(
    processing_graph_specification={
        STEP_DATASET_A: {"input_type": "dataset", "provides_dataset_config_names": True},
        STEP_CONFIG_B: {"input_type": "config", "provides_config_split_names": True, "triggered_by": STEP_DATASET_A},
        STEP_SPLIT_C: {"input_type": "split", "triggered_by": STEP_CONFIG_B},
    }
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


JOB_RUNNER_VERSION = 1


def get_dataset_backfill_plan(
    processing_graph: ProcessingGraph,
    dataset: str = DATASET_NAME,
    revision: str = REVISION_NAME,
    error_codes_to_retry: Optional[List[str]] = None,
) -> DatasetBackfillPlan:
    return DatasetBackfillPlan(
        dataset=dataset,
        revision=revision,
        processing_graph=processing_graph,
        error_codes_to_retry=error_codes_to_retry,
    )


def assert_equality(value: Any, expected: Any, context: Optional[str] = None) -> None:
    report = {"expected": expected, "got": value}
    if context is not None:
        report["additional"] = context
    assert value == expected, report


def assert_dataset_backfill_plan(
    dataset_backfill_plan: DatasetBackfillPlan,
    cache_status: Dict[str, List[str]],
    queue_status: Dict[str, List[str]],
    tasks: List[str],
    config_names: Optional[List[str]] = None,
    split_names_in_first_config: Optional[List[str]] = None,
) -> None:
    if config_names is not None:
        assert_equality(dataset_backfill_plan.dataset_state.config_names, config_names, context="config_names")
        assert_equality(
            len(dataset_backfill_plan.dataset_state.config_states), len(config_names), context="config_states"
        )
        if len(config_names) and split_names_in_first_config is not None:
            assert_equality(
                dataset_backfill_plan.dataset_state.config_states[0].split_names,
                split_names_in_first_config,
                context="split_names",
            )
    computed_cache_status = dataset_backfill_plan.cache_status.as_response()
    for key, value in cache_status.items():
        assert_equality(computed_cache_status[key], sorted(value), key)
    assert_equality(
        dataset_backfill_plan.get_queue_status().as_response(),
        {key: sorted(value) for key, value in queue_status.items()},
        context="queue_status",
    )
    assert_equality(dataset_backfill_plan.as_response(), sorted(tasks), context="tasks")


def put_cache(
    step: str,
    dataset: str,
    revision: str,
    config: Optional[str] = None,
    split: Optional[str] = None,
    error_code: Optional[str] = None,
    use_old_job_runner_version: Optional[bool] = False,
) -> None:
    if not config:
        if not step.startswith("dataset-"):
            raise ValueError("Unexpected artifact: should start with dataset-")
        content = CONFIG_NAMES_CONTENT
        config = None
        split = None
    elif not split:
        if not step.startswith("config-"):
            raise ValueError("Unexpected artifact: should start with config-")
        content = SPLIT_NAMES_CONTENT
        split = None
    else:
        if not step.startswith("split-"):
            raise ValueError("Unexpected artifact: should start with split-")
        content = {}

    if error_code:
        http_status = HTTPStatus.INTERNAL_SERVER_ERROR
        content = {}
    else:
        http_status = HTTPStatus.OK

    upsert_response(
        kind=step,
        dataset=dataset,
        config=config,
        split=split,
        content=content,
        http_status=http_status,
        job_runner_version=JOB_RUNNER_VERSION - 1 if use_old_job_runner_version else JOB_RUNNER_VERSION,
        dataset_git_revision=revision,
        error_code=error_code,
    )


def process_next_job() -> None:
    job_info = Queue().start_job()
    put_cache(
        step=job_info["type"],
        dataset=job_info["params"]["dataset"],
        revision=job_info["params"]["revision"],
        config=job_info["params"]["config"],
        split=job_info["params"]["split"],
    )
    Queue().finish_job(job_id=job_info["job_id"], is_success=True)


def process_all_jobs() -> None:
    runs = 100
    try:
        while runs > 0:
            runs -= 1
            process_next_job()
    except Exception:
        return


def compute_all(
    processing_graph: ProcessingGraph,
    dataset: str = DATASET_NAME,
    revision: str = REVISION_NAME,
    error_codes_to_retry: Optional[List[str]] = None,
) -> None:
    dataset_backfill_plan = get_dataset_backfill_plan(processing_graph, dataset, revision, error_codes_to_retry)
    max_runs = 100
    while len(dataset_backfill_plan.tasks) > 0 and max_runs >= 0:
        if max_runs == 0:
            raise ValueError("Too many runs")
        max_runs -= 1
        dataset_backfill_plan.run()
        for task in dataset_backfill_plan.tasks:
            task_type, sep, num = task.id.partition(",")
            if sep is None:
                raise ValueError(f"Unexpected task id {task.id}: should contain a comma")
            if task_type == "CreateJobs":
                process_all_jobs()
        dataset_backfill_plan = get_dataset_backfill_plan(processing_graph, dataset, revision, error_codes_to_retry)


def artifact_id_to_job_info(artifact_id: str) -> JobInfo:
    dataset, revision, config, split, processing_step_name = Artifact.parse_id(artifact_id)
    return JobInfo(
        job_id="job_id",
        params={
            "dataset": dataset,
            "config": config,
            "split": split,
            "revision": revision,
        },
        type=processing_step_name,
        priority=Priority.NORMAL,
    )
