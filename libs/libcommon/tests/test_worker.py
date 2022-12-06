from typing import Any, Mapping, Optional

import pytest

from libcommon.config import CommonConfig, QueueConfig, WorkerConfig
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
    def get_dataset_git_revision(
        self,
        dataset: str,
        hf_endpoint: str,
        hf_token: Optional[str] = None,
    ) -> Optional[str]:
        return "0.1.2"

    def compute(
        self, dataset: str, config: Optional[str] = None, split: Optional[str] = None, force: bool = False
    ) -> Mapping[str, Any]:
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
    "worker_version, other_version, expected, should_raise",
    [
        ("1.0.0", "1.0.1", 0, False),
        ("1.0.0", "2.0.1", -1, False),
        ("2.0.0", "1.0.1", 1, False),
        ("not a version", "1.0.1", None, True),
    ],
)
def test_compare_major_version(
    test_processing_step: ProcessingStep,
    common_config: CommonConfig,
    queue_config: QueueConfig,
    worker_config: WorkerConfig,
    worker_version: str,
    other_version: str,
    expected: int,
    should_raise: bool,
) -> None:
    worker = DummyWorker(
        processing_step=test_processing_step,
        common_config=common_config,
        queue_config=queue_config,
        worker_config=worker_config,
        version=worker_version,
    )
    if should_raise:
        with pytest.raises(Exception):
            worker.compare_major_version(other_version)
    else:
        assert worker.compare_major_version(other_version) == expected


def test_should_skip_job(
    test_processing_step: ProcessingStep,
    common_config: CommonConfig,
    queue_config: QueueConfig,
    worker_config: WorkerConfig,
) -> None:
    worker = DummyWorker(
        processing_step=test_processing_step,
        common_config=common_config,
        queue_config=queue_config,
        worker_config=worker_config,
        version="1.0.0",
    )
    dataset = "dataset"
    config = "config"
    split = "split"

    assert worker.should_skip_job(dataset=dataset, config=config, split=split) is False
    # we add an entry to the cache
    worker.process(dataset=dataset, config=config, split=split)
    assert worker.should_skip_job(dataset=dataset, config=config, split=split) is True

    assert worker.should_skip_job(dataset=dataset) is False
    # we add an entry to the cache
    worker.process(dataset=dataset)
    assert worker.should_skip_job(dataset=dataset) is True
