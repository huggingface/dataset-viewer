# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus

import pytest

from libcommon.config import ProcessingGraphConfig
from libcommon.exceptions import DatasetInBlockListError
from libcommon.orchestrator import AfterJobPlan, DatasetOrchestrator
from libcommon.processing_graph import Artifact, ProcessingGraph
from libcommon.queue import JobDocument, Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import (
    CachedResponseDocument,
    has_some_cache,
    upsert_response_params,
)
from libcommon.utils import JobOutput, JobResult, Priority, Status

from .utils import (
    ARTIFACT_CA_1,
    ARTIFACT_CA_2,
    ARTIFACT_DA,
    ARTIFACT_DB,
    ARTIFACT_DC,
    ARTIFACT_DD,
    ARTIFACT_DE,
    ARTIFACT_DG,
    ARTIFACT_DH,
    CONFIG_NAMES_CONTENT,
    DATASET_NAME,
    DIFFICULTY,
    JOB_RUNNER_VERSION,
    PROCESSING_GRAPH_FAN_IN_OUT,
    PROCESSING_GRAPH_GENEALOGY,
    PROCESSING_GRAPH_ONE_STEP,
    PROCESSING_GRAPH_PARALLEL,
    REVISION_NAME,
    STEP_CB,
    STEP_DA,
    STEP_DC,
    STEP_DD,
    artifact_id_to_job_info,
)

CACHE_MAX_DAYS = 90


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
    artifacts_to_create: list[str],
) -> None:
    job_info = artifact_id_to_job_info(ARTIFACT_DA)
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
    job_info = artifact_id_to_job_info(ARTIFACT_DA)
    # create two jobs for DG, and none for DH
    # one job should be deleted for DG, and one should be created for DH
    Queue().create_jobs([artifact_id_to_job_info(ARTIFACT_DG)] * 2)

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
    "processing_graph,artifacts_to_create",
    [
        (PROCESSING_GRAPH_ONE_STEP, []),
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DC]),
        (PROCESSING_GRAPH_FAN_IN_OUT, [ARTIFACT_CA_1, ARTIFACT_CA_2]),
        (PROCESSING_GRAPH_PARALLEL, [ARTIFACT_DG, ARTIFACT_DH]),
    ],
)
def test_finish_job(
    processing_graph: ProcessingGraph,
    artifacts_to_create: list[str],
) -> None:
    Queue().add_job(
        dataset=DATASET_NAME,
        revision=REVISION_NAME,
        config=None,
        split=None,
        job_type=STEP_DA,
        priority=Priority.NORMAL,
        difficulty=DIFFICULTY,
    )
    job_info = Queue().start_job()
    job_result = JobResult(
        job_info=job_info,
        job_runner_version=JOB_RUNNER_VERSION,
        is_success=True,
        output=JobOutput(
            content=CONFIG_NAMES_CONTENT,
            http_status=HTTPStatus.OK,
            error_code=None,
            details=None,
            progress=1.0,
        ),
    )
    dataset_orchestrator = DatasetOrchestrator(
        dataset=DATASET_NAME, processing_graph=processing_graph, blocked_datasets=[]
    )
    dataset_orchestrator.finish_job(job_result=job_result)

    assert JobDocument.objects(dataset=DATASET_NAME).count() == 1 + len(artifacts_to_create)

    done_job = JobDocument.objects(dataset=DATASET_NAME, status=Status.SUCCESS)
    assert done_job.count() == 1

    waiting_jobs = JobDocument.objects(dataset=DATASET_NAME, status=Status.WAITING)
    assert waiting_jobs.count() == len(artifacts_to_create)
    assert {job.type for job in waiting_jobs} == {Artifact.parse_id(artifact)[4] for artifact in artifacts_to_create}

    assert CachedResponseDocument.objects(dataset=DATASET_NAME).count() == 1
    cached_response = CachedResponseDocument.objects(dataset=DATASET_NAME).first()
    assert cached_response
    assert cached_response.content == CONFIG_NAMES_CONTENT
    assert cached_response.http_status == HTTPStatus.OK
    assert cached_response.error_code is None
    assert cached_response.details == {}
    assert cached_response.progress == 1.0
    assert cached_response.job_runner_version == JOB_RUNNER_VERSION
    assert cached_response.dataset_git_revision == REVISION_NAME

    # check for blocked dataset
    dataset_orchestrator = DatasetOrchestrator(
        dataset=DATASET_NAME, processing_graph=processing_graph, blocked_datasets=[DATASET_NAME]
    )
    with pytest.raises(DatasetInBlockListError):
        dataset_orchestrator.finish_job(job_result=job_result)
    pending_jobs_df = Queue().get_pending_jobs_df(dataset=DATASET_NAME)
    assert len(pending_jobs_df) == 0
    assert has_some_cache(dataset=DATASET_NAME) is False


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
    first_artifacts: list[str],
) -> None:
    dataset_orchestrator = DatasetOrchestrator(
        dataset=DATASET_NAME, processing_graph=processing_graph, blocked_datasets=[]
    )

    dataset_orchestrator.set_revision(
        revision=REVISION_NAME, priority=Priority.NORMAL, error_codes_to_retry=[], cache_max_days=CACHE_MAX_DAYS
    )

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

    # check for blocked dataset
    dataset_orchestrator = DatasetOrchestrator(
        dataset=DATASET_NAME, processing_graph=processing_graph, blocked_datasets=[DATASET_NAME]
    )
    with pytest.raises(DatasetInBlockListError):
        dataset_orchestrator.set_revision(
            revision=REVISION_NAME, priority=Priority.NORMAL, error_codes_to_retry=[], cache_max_days=CACHE_MAX_DAYS
        )
    pending_jobs_df = Queue().get_pending_jobs_df(dataset=DATASET_NAME)
    assert len(pending_jobs_df) == 0
    assert has_some_cache(dataset=DATASET_NAME) is False


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
    first_artifacts: list[str],
) -> None:
    # create two pending jobs for DA
    Queue().create_jobs([artifact_id_to_job_info(ARTIFACT_DA)] * 2)
    dataset_orchestrator = DatasetOrchestrator(
        dataset=DATASET_NAME, processing_graph=processing_graph, blocked_datasets=[]
    )
    dataset_orchestrator.set_revision(
        revision=REVISION_NAME, priority=Priority.NORMAL, error_codes_to_retry=[], cache_max_days=CACHE_MAX_DAYS
    )

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
    "processing_graph,pending_artifacts,processing_step_names,expected_has_pending_ancestor_jobs",
    [
        (PROCESSING_GRAPH_ONE_STEP, [ARTIFACT_DA], [STEP_DA], True),
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DA, ARTIFACT_DB], [STEP_DA], True),
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DB], [STEP_DD], True),
        (PROCESSING_GRAPH_GENEALOGY, [ARTIFACT_DD], [STEP_DC], False),
        (PROCESSING_GRAPH_FAN_IN_OUT, [ARTIFACT_DA], [STEP_CB], True),
        (PROCESSING_GRAPH_FAN_IN_OUT, [ARTIFACT_DE], [STEP_CB], False),
    ],
)
def test_has_pending_ancestor_jobs(
    processing_graph: ProcessingGraph,
    pending_artifacts: list[str],
    processing_step_names: list[str],
    expected_has_pending_ancestor_jobs: bool,
) -> None:
    Queue().create_jobs([artifact_id_to_job_info(artifact) for artifact in pending_artifacts])
    dataset_orchestrator = DatasetOrchestrator(
        dataset=DATASET_NAME, processing_graph=processing_graph, blocked_datasets=[]
    )
    assert dataset_orchestrator.has_pending_ancestor_jobs(processing_step_names) == expected_has_pending_ancestor_jobs


def test_remove_dataset() -> None:
    Queue().create_jobs([artifact_id_to_job_info(artifact) for artifact in [ARTIFACT_DA, ARTIFACT_DB]])
    Queue().start_job()
    job_info = artifact_id_to_job_info(ARTIFACT_DA)
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

    pending_jobs_df = Queue().get_pending_jobs_df(dataset=DATASET_NAME)
    assert len(pending_jobs_df) > 0
    assert has_some_cache(dataset=DATASET_NAME) is True

    dataset_orchestrator = DatasetOrchestrator(
        dataset=DATASET_NAME, processing_graph=PROCESSING_GRAPH_GENEALOGY, blocked_datasets=[]
    )
    dataset_orchestrator.remove_dataset()

    pending_jobs_df = Queue().get_pending_jobs_df(dataset=DATASET_NAME)
    assert len(pending_jobs_df) == 0
    assert has_some_cache(dataset=DATASET_NAME) is False


@pytest.mark.parametrize("is_big", [False, True])
def test_after_job_plan_gives_bonus_difficulty(is_big: bool) -> None:
    bonus_difficulty_if_dataset_is_big = 10
    processing_graph = ProcessingGraph(
        ProcessingGraphConfig(
            {
                "dataset_step": {"input_type": "dataset"},
                "config_with_split_names_step": {
                    "input_type": "config",
                    "provides_config_split_names": True,
                    "triggered_by": "dataset_step",
                },
                "config_info_step": {
                    "input_type": "config",
                    "provides_config_info": True,
                    "triggered_by": "dataset_step",
                },
                "config_step_with_bonus": {
                    "input_type": "config",
                    "bonus_difficulty_if_dataset_is_big": bonus_difficulty_if_dataset_is_big,
                    "triggered_by": "config_info_step",
                },
                "split_step_with_bonus": {
                    "input_type": "split",
                    "bonus_difficulty_if_dataset_is_big": bonus_difficulty_if_dataset_is_big,
                    "triggered_by": "config_info_step",
                },
            },
            min_bytes_for_bonus_difficulty=1000,
        )
    )
    job_info = artifact_id_to_job_info("config_with_split_names_step,dataset_name,revision_hash,config_name")
    upsert_response_params(
        # inputs
        kind=job_info["type"],
        job_params=job_info["params"],
        job_runner_version=1,
        # output
        content={"splits": [{"dataset_name": "dataset_name", "config": "config_name", "split": "split_name"}]},
        http_status=HTTPStatus.OK,
        error_code=None,
        details=None,
        progress=1.0,
    )
    job_info = artifact_id_to_job_info("config_info_step,dataset_name,revision_hash,config_name")
    upsert_response_params(
        # inputs
        kind=job_info["type"],
        job_params=job_info["params"],
        job_runner_version=1,
        # output
        content={"dataset_info": {"dataset_size": 10000 if is_big else 10}},
        http_status=HTTPStatus.OK,
        error_code=None,
        details=None,
        progress=1.0,
    )

    after_job_plan = AfterJobPlan(job_info=job_info, processing_graph=processing_graph)
    assert len(after_job_plan.job_infos_to_create) == 2
    config_jobs_with_bonus = [
        job for job in after_job_plan.job_infos_to_create if job["type"] == "config_step_with_bonus"
    ]
    assert len(config_jobs_with_bonus) == 1
    split_jobs_with_bonus = [
        job for job in after_job_plan.job_infos_to_create if job["type"] == "split_step_with_bonus"
    ]
    assert len(split_jobs_with_bonus) == 1

    if is_big:
        assert config_jobs_with_bonus[0]["difficulty"] == DIFFICULTY + bonus_difficulty_if_dataset_is_big
        assert split_jobs_with_bonus[0]["difficulty"] == DIFFICULTY + bonus_difficulty_if_dataset_is_big
    else:
        assert config_jobs_with_bonus[0]["difficulty"] == DIFFICULTY
        assert split_jobs_with_bonus[0]["difficulty"] == DIFFICULTY
