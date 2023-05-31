# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus
from typing import List

import pytest

from libcommon.orchestrator import AfterJobPlan, DatasetOrchestrator
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import upsert_response_params
from libcommon.state import Artifact
from libcommon.utils import JobInfo, Priority

from .utils import (
    ARTIFACT_CA_1,
    ARTIFACT_CA_2,
    ARTIFACT_DA,
    ARTIFACT_DB,
    ARTIFACT_DC,
    ARTIFACT_DG,
    ARTIFACT_DH,
    CONFIG_NAMES_CONTENT,
    DATASET_NAME,
    JOB_RUNNER_VERSION,
    PROCESSING_GRAPH_FAN_IN_OUT,
    PROCESSING_GRAPH_GENEALOGY,
    PROCESSING_GRAPH_ONE_STEP,
    PROCESSING_GRAPH_PARALLEL,
    REVISION_NAME,
    STEP_DA,
    STEP_DG,
)


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


@pytest.fixture(autouse=True)
def cache_mongo_resource_autouse(cache_mongo_resource: CacheMongoResource) -> CacheMongoResource:
    return cache_mongo_resource


@pytest.mark.parametrize(
    "processing_graph,artifacts_to_create",
    [
        (PROCESSING_GRAPH_ONE_STEP, []),
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DC]),
        (PROCESSING_GRAPH_FAN_IN_OUT, [ARTIFACT_CA_1, ARTIFACT_CA_2]),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DG, ARTIFACT_DH]),
    ],
)
def test_after_job_plan(
    processing_graph: ProcessingGraph,
    artifacts_to_create: List[str],
) -> None:
    job_info: JobInfo = {
        "job_id": "job_id",
        "params": {
            "dataset": DATASET_NAME,
            "config": None,
            "split": None,
            "revision": REVISION_NAME,
        },
        "type": STEP_DA,
        "priority": Priority.NORMAL,
    }
    # put the cache (to be able to get the config names - case PROCESSING_GRAPH_FAN_IN_OUT)
    upsert_response_params(
        # inputs
        kind=STEP_DA,
        job_params=job_info["params"],
        job_runner_version=JOB_RUNNER_VERSION,
        # output
        content=CONFIG_NAMES_CONTENT,
        http_status=HTTPStatus.OK,
        error_code=None,
        details=None,
        progress=1.0,
    )
    after_job_plan = AfterJobPlan(
        processing_graph=processing_graph,
        job_info=job_info,
    )
    if len(artifacts_to_create):
        assert after_job_plan.as_response() == [f"CreateJobs,{len(artifacts_to_create)}"]
    else:
        assert after_job_plan.as_response() == []

    after_job_plan.run()
    pending_jobs_df = Queue().get_pending_jobs_df(dataset=DATASET_NAME)
    assert len(pending_jobs_df) == len(artifacts_to_create)
    artifact_ids = [
        Artifact.get_id(
            dataset=row["dataset"],
            revision=row["revision"],
            config=row["config"],
            split=row["split"],
            processing_step_name=row["type"],
        )
        for _, row in pending_jobs_df.iterrows()
    ]
    assert set(artifact_ids) == set(artifacts_to_create)


def test_after_job_plan_delete() -> None:
    job_info: JobInfo = {
        "job_id": "job_id",
        "params": {
            "dataset": DATASET_NAME,
            "config": None,
            "split": None,
            "revision": REVISION_NAME,
        },
        "type": STEP_DA,
        "priority": Priority.NORMAL,
    }
    # create two jobs for DG, and none for DH
    # one job should be deleted for DG, and one should be created for DH
    Queue().create_jobs(
        [
            {
                "job_id": "job_id",
                "params": {
                    "dataset": DATASET_NAME,
                    "config": None,
                    "split": None,
                    "revision": REVISION_NAME,
                },
                "type": STEP_DG,
                "priority": Priority.NORMAL,
            }
        ]
        * 2
    )

    after_job_plan = AfterJobPlan(
        processing_graph=PROCESSING_GRAPH_PARALLEL,
        job_info=job_info,
    )
    assert after_job_plan.as_response() == ["CreateJobs,1", "DeleteJobs,1"]

    after_job_plan.run()
    pending_jobs_df = Queue().get_pending_jobs_df(dataset=DATASET_NAME)
    assert len(pending_jobs_df) == 2
    artifact_ids = [
        Artifact.get_id(
            dataset=row["dataset"],
            revision=row["revision"],
            config=row["config"],
            split=row["split"],
            processing_step_name=row["type"],
        )
        for _, row in pending_jobs_df.iterrows()
    ]
    assert artifact_ids == [ARTIFACT_DG, ARTIFACT_DH]


@pytest.mark.parametrize(
    "processing_graph,first_artifacts",
    [
        (PROCESSING_GRAPH_ONE_STEP, [ARTIFACT_DA]),
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DA, ARTIFACT_DB]),
        (PROCESSING_GRAPH_FAN_IN_OUT, [ARTIFACT_DA]),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DA]),
    ],
)
def test_set_revision(
    processing_graph: ProcessingGraph,
    first_artifacts: List[str],
) -> None:
    dataset_orchestrator = DatasetOrchestrator(dataset=DATASET_NAME, processing_graph=processing_graph)

    dataset_orchestrator.set_revision(revision=REVISION_NAME, priority=Priority.NORMAL, error_codes_to_retry=[])

    pending_jobs_df = Queue().get_pending_jobs_df(dataset=DATASET_NAME)
    assert len(pending_jobs_df) == len(first_artifacts)
    artifact_ids = [
        Artifact.get_id(
            dataset=row["dataset"],
            revision=row["revision"],
            config=row["config"],
            split=row["split"],
            processing_step_name=row["type"],
        )
        for _, row in pending_jobs_df.iterrows()
    ]
    assert set(artifact_ids) == set(first_artifacts)


@pytest.mark.parametrize(
    "processing_graph,first_artifacts",
    [
        (PROCESSING_GRAPH_ONE_STEP, [ARTIFACT_DA]),
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DA, ARTIFACT_DB]),
        (PROCESSING_GRAPH_FAN_IN_OUT, [ARTIFACT_DA]),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DA]),
    ],
)
def test_set_revision_handle_existing_jobs(
    processing_graph: ProcessingGraph,
    first_artifacts: List[str],
) -> None:
    # create two pending jobs for DA
    Queue().create_jobs(
        [
            {
                "job_id": "job_id",
                "params": {
                    "dataset": DATASET_NAME,
                    "config": None,
                    "split": None,
                    "revision": REVISION_NAME,
                },
                "type": STEP_DA,
                "priority": Priority.NORMAL,
            }
        ]
        * 2
    )

    dataset_orchestrator = DatasetOrchestrator(dataset=DATASET_NAME, processing_graph=processing_graph)
    dataset_orchestrator.set_revision(revision=REVISION_NAME, priority=Priority.NORMAL, error_codes_to_retry=[])

    pending_jobs_df = Queue().get_pending_jobs_df(dataset=DATASET_NAME)
    assert len(pending_jobs_df) == len(first_artifacts)
    artifact_ids = [
        Artifact.get_id(
            dataset=row["dataset"],
            revision=row["revision"],
            config=row["config"],
            split=row["split"],
            processing_step_name=row["type"],
        )
        for _, row in pending_jobs_df.iterrows()
    ]
    assert set(artifact_ids) == set(first_artifacts)
