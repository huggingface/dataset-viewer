from dataclasses import dataclass
from http import HTTPStatus
from typing import Optional
from unittest.mock import Mock

import pytest
from libcommon.config import CommonConfig
from libcommon.exceptions import CustomError
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Priority, Queue, Status
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import (
    CachedResponse,
    get_response_with_details,
    upsert_response,
)

from worker.config import WorkerConfig
from worker.job_runner import (
    ERROR_CODES_TO_RETRY,
    CompleteJobResult,
    JobRunner,
    PreviousStepError,
)


@pytest.fixture(autouse=True)
def prepare_and_clean_mongo(
    cache_mongo_resource: CacheMongoResource,
    queue_mongo_resource: QueueMongoResource,
) -> None:
    # prepare the database before each test, and clean it afterwards
    pass


class DummyJobRunner(JobRunner):
    # override get_dataset_git_revision to avoid making a request to the Hub
    def get_dataset_git_revision(self) -> Optional[str]:
        return DummyJobRunner._get_dataset_git_revision()

    @staticmethod
    def _get_dataset_git_revision() -> Optional[str]:
        return "0.1.2"

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


@pytest.mark.parametrize(
    "cache_entry,expected_skip",
    [
        (
            CacheEntry(
                error_code="DoNotRetry",  # an error that we don't want to retry
                job_runner_version=DummyJobRunner.get_job_runner_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            True,  # skip
        ),
        (
            CacheEntry(
                error_code=None,  # no error
                job_runner_version=DummyJobRunner.get_job_runner_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            True,  # skip
        ),
        (
            None,  # no cache entry
            False,  # process
        ),
        (
            CacheEntry(
                error_code=ERROR_CODES_TO_RETRY[0],  # an error that we want to retry
                job_runner_version=DummyJobRunner.get_job_runner_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            False,  # process
        ),
        (
            CacheEntry(
                error_code="DoNotRetry",
                job_runner_version=None,  # no version
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            False,  # process
        ),
        (
            CacheEntry(
                error_code="DoNotRetry",
                job_runner_version=0,  # a different version
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            False,  # process
        ),
        (
            CacheEntry(
                error_code="DoNotRetry",
                job_runner_version=DummyJobRunner.get_job_runner_version(),
                dataset_git_revision=None,  # no dataset git revision
            ),
            False,  # process
        ),
        (
            CacheEntry(
                error_code="DoNotRetry",
                job_runner_version=DummyJobRunner.get_job_runner_version(),
                dataset_git_revision="different",  # a different dataset git revision
            ),
            False,  # process
        ),
        (
            CacheEntry(
                error_code=None,  # no error
                job_runner_version=DummyJobRunner.get_job_runner_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
                progress=0.5,  # incomplete result
            ),
            False,  # process
        ),
        (
            CacheEntry(
                error_code=None,  # no error
                job_runner_version=DummyJobRunner.get_job_runner_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
                progress=1.0,  # complete result
            ),
            True,  # skip
        ),
    ],
)
def test_should_skip_job(
    test_processing_graph: ProcessingGraph,
    test_processing_step: ProcessingStep,
    cache_entry: Optional[CacheEntry],
    expected_skip: bool,
) -> None:
    job_id = "job_id"
    dataset = "dataset"
    config = "config"
    split = "split"
    job_runner = DummyJobRunner(
        job_info={
            "job_id": job_id,
            "type": test_processing_step.job_type,
            "dataset": dataset,
            "config": config,
            "split": split,
            "priority": Priority.NORMAL,
        },
        processing_step=test_processing_step,
        processing_graph=test_processing_graph,
        common_config=CommonConfig(),
        worker_config=WorkerConfig(),
    )
    if cache_entry:
        upsert_response(
            kind=test_processing_step.cache_kind,
            dataset=dataset,
            config=config,
            split=split,
            content={},
            http_status=HTTPStatus.OK,  # <- not important
            error_code=cache_entry.error_code,
            details=None,
            job_runner_version=cache_entry.job_runner_version,
            dataset_git_revision=cache_entry.dataset_git_revision,
            progress=cache_entry.progress,
        )
    assert job_runner.should_skip_job() is expected_skip


def test_check_type(
    test_processing_graph: ProcessingGraph,
    another_processing_step: ProcessingStep,
    test_processing_step: ProcessingStep,
) -> None:
    job_id = "job_id"
    dataset = "dataset"
    config = "config"
    split = "split"

    job_type = f"not-{test_processing_step.job_type}"
    with pytest.raises(ValueError):
        DummyJobRunner(
            job_info={
                "job_id": job_id,
                "type": job_type,
                "dataset": dataset,
                "config": config,
                "split": split,
                "priority": Priority.NORMAL,
            },
            processing_step=test_processing_step,
            processing_graph=test_processing_graph,
            common_config=CommonConfig(),
            worker_config=WorkerConfig(),
        )
    with pytest.raises(ValueError):
        DummyJobRunner(
            job_info={
                "job_id": job_id,
                "type": test_processing_step.job_type,
                "dataset": dataset,
                "config": config,
                "split": split,
                "priority": Priority.NORMAL,
            },
            processing_step=another_processing_step,
            processing_graph=test_processing_graph,
            common_config=CommonConfig(),
            worker_config=WorkerConfig(),
        )


@pytest.mark.parametrize(
    "priority",
    [
        Priority.LOW,
        Priority.NORMAL,
    ],
)
def test_backfill(priority: Priority) -> None:
    graph = ProcessingGraph(
        {
            "dummy": {"input_type": "dataset"},
            "dataset-child": {"input_type": "dataset", "triggered_by": "dummy"},
            "config-child": {"input_type": "config", "triggered_by": "dummy"},
            "dataset-unrelated": {"input_type": "dataset"},
        }
    )
    root_step = graph.get_processing_step("dummy")
    job_runner = DummyJobRunner(
        job_info={
            "job_id": "job_id",
            "type": root_step.job_type,
            "dataset": "dataset",
            "config": None,
            "split": None,
            "priority": priority,
        },
        processing_step=root_step,
        processing_graph=graph,
        common_config=CommonConfig(),
        worker_config=WorkerConfig(),
    )
    assert not job_runner.should_skip_job()
    # we add an entry to the cache
    job_runner.run()
    assert job_runner.should_skip_job()
    # check that the missing cache entries have been created
    queue = Queue()
    dataset_child_jobs = queue.get_dump_with_status(job_type="dataset-child", status=Status.WAITING)
    assert len(dataset_child_jobs) == 1
    assert dataset_child_jobs[0]["dataset"] == "dataset"
    assert dataset_child_jobs[0]["config"] is None
    assert dataset_child_jobs[0]["split"] is None
    assert dataset_child_jobs[0]["priority"] is priority.value
    dataset_unrelated_jobs = queue.get_dump_with_status(job_type="dataset-unrelated", status=Status.WAITING)
    assert len(dataset_unrelated_jobs) == 1
    assert dataset_unrelated_jobs[0]["dataset"] == "dataset"
    assert dataset_unrelated_jobs[0]["config"] is None
    assert dataset_unrelated_jobs[0]["split"] is None
    assert dataset_unrelated_jobs[0]["priority"] is priority.value
    # check that no config level jobs have been created, because the config names are not known
    config_child_jobs = queue.get_dump_with_status(job_type="config-child", status=Status.WAITING)
    assert len(config_child_jobs) == 0


def test_job_runner_set_crashed(
    test_processing_graph: ProcessingGraph,
    test_processing_step: ProcessingStep,
) -> None:
    job_id = "job_id"
    dataset = "dataset"
    config = "config"
    split = "split"
    message = "I'm crashed :("
    job_runner = DummyJobRunner(
        job_info={
            "job_id": job_id,
            "type": test_processing_step.job_type,
            "dataset": dataset,
            "config": config,
            "split": split,
            "priority": Priority.NORMAL,
        },
        processing_step=test_processing_step,
        processing_graph=test_processing_graph,
        common_config=CommonConfig(),
        worker_config=WorkerConfig(),
    )
    job_runner.set_crashed(message=message)
    response = CachedResponse.objects()[0]
    expected_error = {"error": message}
    assert response.http_status == HTTPStatus.NOT_IMPLEMENTED
    assert response.error_code == "JobRunnerCrashedError"
    assert response.dataset == dataset
    assert response.config == config
    assert response.split == split
    assert response.content == expected_error
    assert response.details == expected_error
    # TODO: check if it stores the correct dataset git sha and job version when it's implemented


def test_raise_if_parallel_response_exists(
    test_processing_graph: ProcessingGraph,
    test_processing_step: ProcessingStep,
) -> None:
    dataset = "dataset"
    config = "config"
    split = "split"
    current_dataset_git_revision = "CURRENT_GIT_REVISION"
    upsert_response(
        kind="dummy-parallel",
        dataset=dataset,
        config=config,
        split=split,
        content={},
        dataset_git_revision=current_dataset_git_revision,
        job_runner_version=1,
        progress=1.0,
        http_status=HTTPStatus.OK,
    )
    job_runner = DummyJobRunner(
        job_info={
            "job_id": "job_id",
            "type": "dummy",
            "dataset": dataset,
            "config": config,
            "split": split,
            "priority": Priority.NORMAL,
        },
        processing_step=test_processing_step,
        processing_graph=test_processing_graph,
        common_config=CommonConfig(),
        worker_config=WorkerConfig(),
    )
    job_runner.get_dataset_git_revision = Mock(return_value=current_dataset_git_revision)  # type: ignore
    with pytest.raises(CustomError) as exc_info:
        job_runner.raise_if_parallel_response_exists(parallel_cache_kind="dummy-parallel", parallel_job_version=1)
    assert exc_info.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert exc_info.value.code == "ResponseAlreadyComputedError"


@pytest.mark.parametrize("disclose_cause", [False, True])
def test_previous_step_error(disclose_cause: bool) -> None:
    dataset = "dataset"
    config = "config"
    split = "split"
    kind = "cache_kind"
    error_code = "ErrorCode"
    error_message = "error message"
    cause_exception = "CauseException"
    cause_message = "cause message"
    cause_traceback = ["traceback1", "traceback2"]
    details = {
        "error": error_message,
        "cause_exception": cause_exception,
        "cause_message": cause_message,
        "cause_traceback": cause_traceback,
    }
    content = details if disclose_cause else {"error": error_message}
    job_runner_version = 1
    dataset_git_revision = "dataset_git_revision"
    progress = 1.0
    upsert_response(
        kind=kind,
        dataset=dataset,
        config=config,
        split=split,
        content=content,
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        error_code=error_code,
        details=details,
        job_runner_version=job_runner_version,
        dataset_git_revision=dataset_git_revision,
        progress=progress,
    )
    response = get_response_with_details(kind=kind, dataset=dataset, config=config, split=split)
    error = PreviousStepError.from_response(response=response, kind=kind, dataset=dataset, config=config, split=split)
    assert error.disclose_cause == disclose_cause
    assert error.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert error.code == error_code
    assert error.as_response_without_cause() == {
        "error": error_message,
    }
    assert error.as_response_with_cause() == {
        "error": error_message,
        "cause_exception": cause_exception,
        "cause_message": cause_message,
        "cause_traceback": [
            "The previous step failed, the error is copied to this step:",
            f"  {kind=} {dataset=} {config=} {split=}",
            "---",
            *cause_traceback,
        ],
    }
    if disclose_cause:
        assert error.as_response() == error.as_response_with_cause()
    else:
        assert error.as_response() == error.as_response_without_cause()
