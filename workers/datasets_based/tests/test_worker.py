from typing import Any, Mapping, Optional

import pytest
from libcommon.config import CommonConfig
from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Queue, Status, _clean_queue_database
from libcommon.simple_cache import SplitFullName, _clean_cache_database

from datasets_based.worker import Worker


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_queue_database()
    _clean_cache_database()


class DummyWorker(Worker):
    # override get_dataset_git_revision to avoid making a request to the Hub
    def get_dataset_git_revision(self) -> Optional[str]:
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
    worker = DummyWorker(
        job_info={
            "job_id": job_id,
            "type": test_processing_step.job_type,
            "dataset": dataset,
            "config": config,
            "split": split,
            "force": force,
        },
        processing_step=test_processing_step,
        common_config=CommonConfig(),
    )
    if should_raise:
        with pytest.raises(Exception):
            worker.compare_major_version(other_version)
    else:
        assert worker.compare_major_version(other_version) == expected


def test_should_skip_job(
    test_processing_step: ProcessingStep,
) -> None:
    job_id = "job_id"
    dataset = "dataset"
    config = "config"
    split = "split"
    force = False
    worker = DummyWorker(
        job_info={
            "job_id": job_id,
            "type": test_processing_step.job_type,
            "dataset": dataset,
            "config": config,
            "split": split,
            "force": force,
        },
        processing_step=test_processing_step,
        common_config=CommonConfig(),
    )
    assert worker.should_skip_job() is False
    # we add an entry to the cache
    worker.process()
    assert worker.should_skip_job() is True


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
        DummyWorker(
            job_info={
                "job_id": job_id,
                "type": job_type,
                "dataset": dataset,
                "config": config,
                "split": split,
                "force": force,
            },
            processing_step=test_processing_step,
            common_config=CommonConfig(),
        )

    another_processing_step = ProcessingStep(
        endpoint=f"not-{test_processing_step.endpoint}",
        input_type="dataset",
        requires=None,
        required_by_dataset_viewer=False,
        parent=None,
        ancestors=[],
        children=[],
    )
    with pytest.raises(ValueError):
        DummyWorker(
            job_info={
                "job_id": job_id,
                "type": test_processing_step.job_type,
                "dataset": dataset,
                "config": config,
                "split": split,
                "force": force,
            },
            processing_step=another_processing_step,
            common_config=CommonConfig(),
        )


def test_create_children_jobs() -> None:
    graph = ProcessingGraph(
        {
            "/dummy": {"input_type": "dataset"},
            "/child-dataset": {"input_type": "dataset", "requires": "/dummy"},
            "/child-split": {"input_type": "split", "requires": "/dummy"},
        }
    )
    root_step = graph.get_step("/dummy")
    worker = DummyWorker(
        job_info={
            "job_id": "job_id",
            "type": root_step.job_type,
            "dataset": "dataset",
            "config": None,
            "split": None,
            "force": False,
        },
        processing_step=root_step,
        common_config=CommonConfig(),
    )
    assert worker.should_skip_job() is False
    # we add an entry to the cache
    worker.process()
    assert worker.should_skip_job() is True
    # check that the children jobs have been created
    child_dataset_jobs = Queue(type="/child-dataset").get_dump_with_status(status=Status.WAITING)
    assert len(child_dataset_jobs) == 1
    assert child_dataset_jobs[0]["dataset"] == "dataset"
    assert child_dataset_jobs[0]["config"] is None
    assert child_dataset_jobs[0]["split"] is None
    child_split_jobs = Queue(type="/child-split").get_dump_with_status(status=Status.WAITING)
    assert len(child_split_jobs) == 2
    assert all(job["dataset"] == "dataset" and job["config"] == "config" for job in child_split_jobs)
    # we don't know the order
    assert {child_split_jobs[0]["split"], child_split_jobs[1]["split"]} == {"split1", "split2"}
