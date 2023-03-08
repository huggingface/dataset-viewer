from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Mapping, Optional

import pytest
from libcommon.config import CommonConfig
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Priority, Queue, Status
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import CachedResponse, SplitFullName, upsert_response

from worker.config import WorkerConfig
from worker.job_runner import ERROR_CODES_TO_RETRY, JobRunner


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
    def get_job_type() -> str:
        return "/dummy"

    @staticmethod
    def get_version() -> str:
        return "1.0.1"

    def compute(self) -> Mapping[str, Any]:
        return {"key": "value"}

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        return {SplitFullName(self.dataset, "config", "split1"), SplitFullName(self.dataset, "config", "split2")}


@pytest.mark.parametrize(
    "other_version, expected, should_raise",
    [
        ("1.0.0", 0, False),
        ("0.1.0", 1, False),
        ("2.0.0", -1, False),
        ("not a version", None, True),
    ],
)
def test_compare_major_version(
    test_processing_step: ProcessingStep,
    other_version: str,
    expected: int,
    should_raise: bool,
) -> None:
    job_id = "job_id"
    dataset = "dataset"
    config = "config"
    split = "split"
    force = False
    job_runner = DummyJobRunner(
        job_info={
            "job_id": job_id,
            "type": test_processing_step.job_type,
            "dataset": dataset,
            "config": config,
            "split": split,
            "force": force,
            "priority": Priority.NORMAL,
        },
        processing_step=test_processing_step,
        common_config=CommonConfig(),
        worker_config=WorkerConfig(),
    )
    if should_raise:
        with pytest.raises(Exception):
            job_runner.compare_major_version(other_version)
    else:
        assert job_runner.compare_major_version(other_version) == expected


@dataclass
class CacheEntry:
    error_code: Optional[str]
    worker_version: Optional[str]
    dataset_git_revision: Optional[str]
    partial: Optional[bool] = None


# .get_version()
@pytest.mark.parametrize(
    "force,cache_entry,expected_skip",
    [
        (
            False,
            CacheEntry(
                error_code="DoNotRetry",  # an error that we don't want to retry
                worker_version=DummyJobRunner.get_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            True,  # skip
        ),
        (
            False,
            CacheEntry(
                error_code=None,  # no error
                worker_version=DummyJobRunner.get_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            True,  # skip
        ),
        (
            True,  # force
            CacheEntry(
                error_code="DoNotRetry",
                worker_version=DummyJobRunner.get_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            False,  # process
        ),
        (
            False,
            None,  # no cache entry
            False,  # process
        ),
        (
            False,
            CacheEntry(
                error_code=ERROR_CODES_TO_RETRY[0],  # an error that we want to retry
                worker_version=DummyJobRunner.get_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            False,  # process
        ),
        (
            False,
            CacheEntry(
                error_code="DoNotRetry",
                worker_version=None,  # no version
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            False,  # process
        ),
        (
            False,
            CacheEntry(
                error_code="DoNotRetry",
                worker_version="0.0.1",  # a different version
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
            ),
            False,  # process
        ),
        (
            False,
            CacheEntry(
                error_code="DoNotRetry",
                worker_version=DummyJobRunner.get_version(),
                dataset_git_revision=None,  # no dataset git revision
            ),
            False,  # process
        ),
        (
            False,
            CacheEntry(
                error_code="DoNotRetry",
                worker_version=DummyJobRunner.get_version(),
                dataset_git_revision="different",  # a different dataset git revision
            ),
            False,  # process
        ),
        (
            False,
            CacheEntry(
                error_code=None,  # no error
                worker_version=DummyJobRunner.get_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
                partial=True,  # incomplete job
            ),
            False,  # process
        ),
        (
            False,
            CacheEntry(
                error_code=None,  # no error
                worker_version=DummyJobRunner.get_version(),
                dataset_git_revision=DummyJobRunner._get_dataset_git_revision(),
                partial=False,  # complete job
            ),
            True,  # skip
        ),
    ],
)
def test_should_skip_job(
    test_processing_step: ProcessingStep, force: bool, cache_entry: Optional[CacheEntry], expected_skip: bool
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
            "force": force,
            "priority": Priority.NORMAL,
        },
        processing_step=test_processing_step,
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
            worker_version=cache_entry.worker_version,
            dataset_git_revision=cache_entry.dataset_git_revision,
            partial=cache_entry.partial,
        )
    assert job_runner.should_skip_job() is expected_skip


def test_check_type(
    test_processing_step: ProcessingStep,
) -> None:
    job_id = "job_id"
    dataset = "dataset"
    config = "config"
    split = "split"
    force = False

    job_type = f"not-{test_processing_step.job_type}"
    with pytest.raises(ValueError):
        DummyJobRunner(
            job_info={
                "job_id": job_id,
                "type": job_type,
                "dataset": dataset,
                "config": config,
                "split": split,
                "force": force,
                "priority": Priority.NORMAL,
            },
            processing_step=test_processing_step,
            common_config=CommonConfig(),
            worker_config=WorkerConfig(),
        )

    another_processing_step = ProcessingStep(
        name=f"not-{test_processing_step.name}",
        input_type="dataset",
        requires=None,
        required_by_dataset_viewer=False,
        parent=None,
        ancestors=[],
        children=[],
    )
    with pytest.raises(ValueError):
        DummyJobRunner(
            job_info={
                "job_id": job_id,
                "type": test_processing_step.job_type,
                "dataset": dataset,
                "config": config,
                "split": split,
                "force": force,
                "priority": Priority.NORMAL,
            },
            processing_step=another_processing_step,
            common_config=CommonConfig(),
            worker_config=WorkerConfig(),
        )


def test_create_children_jobs() -> None:
    graph = ProcessingGraph(
        {
            "/dummy": {"input_type": "dataset"},
            "/child-dataset": {"input_type": "dataset", "requires": "/dummy"},
            "/child-config": {"input_type": "config", "requires": "/dummy"},
            "/child-split": {"input_type": "split", "requires": "/dummy"},
        }
    )
    root_step = graph.get_step("/dummy")
    job_runner = DummyJobRunner(
        job_info={
            "job_id": "job_id",
            "type": root_step.job_type,
            "dataset": "dataset",
            "config": None,
            "split": None,
            "force": False,
            "priority": Priority.LOW,
        },
        processing_step=root_step,
        common_config=CommonConfig(),
        worker_config=WorkerConfig(),
    )
    assert not job_runner.should_skip_job()
    # we add an entry to the cache
    job_runner.run()
    assert job_runner.should_skip_job()
    # check that the children jobs have been created
    queue = Queue()
    child_dataset_jobs = queue.get_dump_with_status(job_type="/child-dataset", status=Status.WAITING)
    assert len(child_dataset_jobs) == 1
    assert child_dataset_jobs[0]["dataset"] == "dataset"
    assert child_dataset_jobs[0]["config"] is None
    assert child_dataset_jobs[0]["split"] is None
    assert child_dataset_jobs[0]["priority"] is Priority.LOW.value
    child_config_jobs = queue.get_dump_with_status(job_type="/child-config", status=Status.WAITING)
    assert len(child_config_jobs) == 1
    assert child_config_jobs[0]["dataset"] == "dataset"
    assert child_config_jobs[0]["config"] == "config"
    assert child_config_jobs[0]["split"] is None
    assert child_config_jobs[0]["priority"] is Priority.LOW.value
    child_split_jobs = queue.get_dump_with_status(job_type="/child-split", status=Status.WAITING)
    assert len(child_split_jobs) == 2
    assert all(
        job["dataset"] == "dataset" and job["config"] == "config" and job["priority"] == Priority.LOW.value
        for job in child_split_jobs
    )
    # we don't know the order
    assert {child_split_jobs[0]["split"], child_split_jobs[1]["split"]} == {"split1", "split2"}


def test_job_runner_set_crashed(
    test_processing_step: ProcessingStep,
) -> None:
    job_id = "job_id"
    dataset = "dataset"
    config = "config"
    split = "split"
    force = False
    message = "I'm crashed :("
    job_runner = DummyJobRunner(
        job_info={
            "job_id": job_id,
            "type": test_processing_step.job_type,
            "dataset": dataset,
            "config": config,
            "split": split,
            "force": force,
            "priority": Priority.NORMAL,
        },
        processing_step=test_processing_step,
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
