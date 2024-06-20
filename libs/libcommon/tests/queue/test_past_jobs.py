# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from datetime import timedelta

import pytest

from libcommon.queue.dataset_blockages import get_blocked_datasets
from libcommon.queue.jobs import Queue
from libcommon.queue.past_jobs import (
    DATASET_BLOCKAGE_THRESHOLD_SECONDS,
    NegativeDurationError,
    PastJobDocument,
    create_past_job,
)
from libcommon.resources import QueueMongoResource
from libcommon.utils import get_datetime


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


DATASET = "dataset"


@pytest.mark.parametrize(
    "duration",
    [1.0, 0.0, 12345.6789],
)
def test_create_past_job(duration: float) -> None:
    finished_at = get_datetime()
    started_at = finished_at - timedelta(seconds=duration)
    create_past_job(dataset=DATASET, started_at=started_at, finished_at=finished_at)
    past_job = PastJobDocument.objects(dataset=DATASET).get()
    assert past_job.duration == duration


@pytest.mark.parametrize("duration", [-1.0])
def test_create_past_job_raises(duration: float) -> None:
    finished_at = get_datetime()
    started_at = finished_at - timedelta(seconds=duration)
    with pytest.raises(NegativeDurationError):
        create_past_job(dataset=DATASET, started_at=started_at, finished_at=finished_at)


def test_create_past_job_raises_if_timezone_unaware() -> None:
    finished_at = get_datetime()

    queue = Queue()
    queue.add_job(job_type="test_type", dataset="test_dataset", revision="test_revision", difficulty=50)
    job_info = queue.start_job()
    started_at = queue._get_started_job(job_info["job_id"]).started_at
    # ^ mongo looses the timezone, see https://github.com/huggingface/dataset-viewer/issues/862
    assert started_at is not None

    with pytest.raises(TypeError) as exc_info:
        create_past_job(dataset=DATASET, started_at=started_at, finished_at=finished_at)
    assert "can't subtract offset-naive and offset-aware datetimes" in str(exc_info.value)


@pytest.mark.parametrize(
    "jobs,expected_blocked_datasets",
    [
        (
            [
                ("dataset1", DATASET_BLOCKAGE_THRESHOLD_SECONDS * 0.5),
                ("dataset2", DATASET_BLOCKAGE_THRESHOLD_SECONDS * 2),
            ],
            ["dataset2"],
        ),
        (
            [
                ("dataset", DATASET_BLOCKAGE_THRESHOLD_SECONDS * 0.9),
                ("dataset", DATASET_BLOCKAGE_THRESHOLD_SECONDS * 0.9),
            ],
            ["dataset"],
        ),
        (
            [
                ("dataset1", DATASET_BLOCKAGE_THRESHOLD_SECONDS * 2),
                ("dataset2", DATASET_BLOCKAGE_THRESHOLD_SECONDS * 2),
            ],
            ["dataset1", "dataset2"],
        ),
    ],
)
def test_block_datasets(jobs: list[tuple[str, float]], expected_blocked_datasets: list[str]) -> None:
    for job in jobs:
        (dataset, duration) = job
        finished_at = get_datetime()
        started_at = finished_at - timedelta(seconds=duration)
        create_past_job(dataset=dataset, started_at=started_at, finished_at=finished_at)
    assert sorted(get_blocked_datasets()) == sorted(expected_blocked_datasets)
