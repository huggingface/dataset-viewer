from typing import Any, Mapping, Optional

import pytest

from libcommon.config import CommonConfig
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import _clean_queue_database
from libcommon.simple_cache import _clean_cache_database
from libcommon.worker import Worker, parse_version


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


@pytest.mark.parametrize(
    "string_version, expected_major_version, should_raise",
    [
        ("1.0.0", 1, False),
        ("3.1.2", 3, False),
        ("1.1", 1, False),
        ("not a version", None, True),
    ],
)
def test_parse_version(string_version: str, expected_major_version: int, should_raise: bool) -> None:
    if should_raise:
        with pytest.raises(Exception):
            parse_version(string_version)
    else:
        assert parse_version(string_version).major == expected_major_version


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
    common_config: CommonConfig,
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
        common_config=common_config,
    )
    if should_raise:
        with pytest.raises(Exception):
            worker.compare_major_version(other_version)
    else:
        assert worker.compare_major_version(other_version) == expected


def test_should_skip_job(
    test_processing_step: ProcessingStep,
    common_config: CommonConfig,
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
        common_config=common_config,
    )
    assert worker.should_skip_job() is False
    # we add an entry to the cache
    worker.process()
    assert worker.should_skip_job() is True


def test_check_type(
    test_processing_step: ProcessingStep,
    common_config: CommonConfig,
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
            common_config=common_config,
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
            common_config=common_config,
        )
