from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional

import pytest
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Job, Queue
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedResponse, get_response, upsert_response
from libcommon.utils import JobInfo, Priority, Status

from worker.config import AppConfig
from worker.job_manager import JobManager
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner
from worker.utils import CompleteJobResult

from .fixtures.hub import get_default_config_split


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


class DummyJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    @staticmethod
    def get_job_type() -> str:
        return "dummy"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult({"key": "value"})


@dataclass
class CacheEntry:
    error_code: Optional[str]
    job_runner_version: Optional[int]
    dataset_git_revision: Optional[str]
    progress: Optional[float] = None


def test_check_type(
    test_processing_graph: ProcessingGraph,
    another_processing_step: ProcessingStep,
    test_processing_step: ProcessingStep,
    app_config: AppConfig,
) -> None:
    job_id = "job_id"
    dataset = "dataset"
    revision = "revision"
    config = "config"
    split = "split"

    job_type = f"not-{test_processing_step.job_type}"
    job_info = JobInfo(
        job_id=job_id,
        type=job_type,
        params={
            "dataset": dataset,
            "revision": revision,
            "config": config,
            "split": split,
        },
        priority=Priority.NORMAL,
    )
    with pytest.raises(ValueError):
        job_runner = DummyJobRunner(
            job_info=job_info,
            processing_step=test_processing_step,
            app_config=app_config,
        )

        JobManager(
            job_info=job_info, app_config=app_config, job_runner=job_runner, processing_graph=test_processing_graph
        )

    job_info = JobInfo(
        job_id=job_id,
        type=test_processing_step.job_type,
        params={
            "dataset": dataset,
            "revision": revision,
            "config": config,
            "split": split,
        },
        priority=Priority.NORMAL,
    )
    with pytest.raises(ValueError):
        job_runner = DummyJobRunner(
            job_info=job_info,
            processing_step=another_processing_step,
            app_config=app_config,
        )

        JobManager(
            job_info=job_info, app_config=app_config, job_runner=job_runner, processing_graph=test_processing_graph
        )


@pytest.mark.parametrize(
    "priority",
    [
        Priority.LOW,
        Priority.NORMAL,
    ],
)
def test_backfill(priority: Priority, app_config: AppConfig) -> None:
    graph = ProcessingGraph(
        {
            "dummy": {"input_type": "dataset"},
            "dataset-child": {"input_type": "dataset", "triggered_by": "dummy"},
            "config-child": {"input_type": "config", "triggered_by": "dummy"},
            "dataset-unrelated": {"input_type": "dataset"},
        }
    )
    root_step = graph.get_processing_step("dummy")
    queue = Queue()
    assert Job.objects().count() == 0
    queue.upsert_job(
        job_type=root_step.job_type,
        dataset="dataset",
        revision="revision",
        config=None,
        split=None,
        priority=priority,
    )
    job_info = queue.start_job()
    assert job_info["priority"] == priority

    job_runner = DummyJobRunner(
        job_info=job_info,
        processing_step=root_step,
        app_config=app_config,
    )

    job_manager = JobManager(job_info=job_info, app_config=app_config, job_runner=job_runner, processing_graph=graph)
    assert job_manager.priority == priority

    job_result = job_manager.run_job()
    assert job_result["is_success"]
    assert job_result["output"] is not None
    assert job_result["output"]["content"] == {"key": "value"}

    job_manager.finish(job_result=job_result)
    # check that the job has been finished
    job = queue.get_job_with_id(job_id=job_info["job_id"])
    assert job.status in [Status.SUCCESS, Status.ERROR, Status.CANCELLED]
    assert job.priority == priority

    # check that the cache entry has have been created
    cached_response = get_response(kind=root_step.cache_kind, dataset="dataset", config=None, split=None)
    assert cached_response is not None
    assert cached_response["http_status"] == HTTPStatus.OK
    assert cached_response["error_code"] is None
    assert cached_response["content"] == {"key": "value"}
    assert cached_response["dataset_git_revision"] == "revision"
    assert cached_response["job_runner_version"] == 1
    assert cached_response["progress"] == 1.0

    dataset_child_jobs = queue.get_dump_with_status(job_type="dataset-child", status=Status.WAITING)
    assert len(dataset_child_jobs) == 1
    assert dataset_child_jobs[0]["dataset"] == "dataset"
    assert dataset_child_jobs[0]["revision"] == "revision"
    assert dataset_child_jobs[0]["config"] is None
    assert dataset_child_jobs[0]["split"] is None
    assert dataset_child_jobs[0]["priority"] is priority.value
    dataset_unrelated_jobs = queue.get_dump_with_status(job_type="dataset-unrelated", status=Status.WAITING)
    assert len(dataset_unrelated_jobs) == 1
    assert dataset_unrelated_jobs[0]["dataset"] == "dataset"
    assert dataset_unrelated_jobs[0]["revision"] == "revision"
    assert dataset_unrelated_jobs[0]["config"] is None
    assert dataset_unrelated_jobs[0]["split"] is None
    assert dataset_unrelated_jobs[0]["priority"] is priority.value
    # check that no config level jobs have been created, because the config names are not known
    config_child_jobs = queue.get_dump_with_status(job_type="config-child", status=Status.WAITING)
    assert len(config_child_jobs) == 0


def test_job_runner_set_crashed(
    test_processing_graph: ProcessingGraph,
    test_processing_step: ProcessingStep,
    app_config: AppConfig,
) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"
    split = "split"
    message = "I'm crashed :("

    queue = Queue()
    assert Job.objects().count() == 0
    queue.upsert_job(
        job_type=test_processing_step.job_type,
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        priority=Priority.NORMAL,
    )
    job_info = queue.start_job()

    job_runner = DummyJobRunner(
        job_info=job_info,
        processing_step=test_processing_step,
        app_config=app_config,
    )

    job_manager = JobManager(
        job_info=job_info, app_config=app_config, job_runner=job_runner, processing_graph=test_processing_graph
    )

    job_manager.set_crashed(message=message)
    response = CachedResponse.objects()[0]
    expected_error = {"error": message}
    assert response.http_status == HTTPStatus.NOT_IMPLEMENTED
    assert response.error_code == "JobManagerCrashedError"
    assert response.dataset == dataset
    assert response.dataset_git_revision == revision
    assert response.config == config
    assert response.split == split
    assert response.content == expected_error
    assert response.details == expected_error
    # TODO: check if it stores the correct dataset git sha and job version when it's implemented


def test_raise_if_parallel_response_exists(
    test_processing_graph: ProcessingGraph,
    test_processing_step: ProcessingStep,
    app_config: AppConfig,
) -> None:
    dataset = "dataset"
    revision = "revision"
    config = "config"
    split = "split"
    upsert_response(
        kind="dummy-parallel",
        dataset=dataset,
        config=config,
        split=split,
        content={},
        dataset_git_revision=revision,
        job_runner_version=1,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )

    job_info = JobInfo(
        job_id="job_id",
        type="dummy",
        params={
            "dataset": dataset,
            "revision": revision,
            "config": config,
            "split": split,
        },
        priority=Priority.NORMAL,
    )
    job_runner = DummyJobRunner(
        job_info=job_info,
        processing_step=test_processing_step,
        app_config=app_config,
    )

    job_manager = JobManager(
        job_info=job_info, app_config=app_config, job_runner=job_runner, processing_graph=test_processing_graph
    )
    with pytest.raises(CustomError) as exc_info:
        job_manager.raise_if_parallel_response_exists(parallel_cache_kind="dummy-parallel", parallel_job_version=1)
    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert exc_info.value.code == "ResponseAlreadyComputedError"


def test_doesnotexist(app_config: AppConfig) -> None:
    dataset = "doesnotexist"
    revision = "revision"
    dataset, config, split = get_default_config_split(dataset)

    job_info = JobInfo(
        job_id="job_id",
        type="dummy",
        params={
            "dataset": dataset,
            "revision": revision,
            "config": config,
            "split": split,
        },
        priority=Priority.NORMAL,
    )
    processing_step_name = "dummy"
    processing_graph = ProcessingGraph(
        {
            "dataset-level": {"input_type": "dataset"},
            processing_step_name: {
                "input_type": "dataset",
                "job_runner_version": DummyJobRunner.get_job_runner_version(),
                "triggered_by": "dataset-level",
            },
        }
    )
    processing_step = processing_graph.get_processing_step(processing_step_name)

    job_runner = DummyJobRunner(
        job_info=job_info,
        processing_step=processing_step,
        app_config=app_config,
    )

    job_manager = JobManager(
        job_info=job_info, app_config=app_config, job_runner=job_runner, processing_graph=processing_graph
    )

    job_result = job_manager.process()
    # ^ the job is processed, since we don't contact the Hub to check if the dataset exists
    assert job_result["output"] is not None
    assert job_result["output"]["content"] == {"key": "value"}
