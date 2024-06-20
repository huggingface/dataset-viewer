# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from datetime import timedelta
from typing import Optional

import pytest

from libcommon.queue.past_jobs import NegativeDurationError, PastJobDocument, create_past_job
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
def test_create_past_job_raises(duration: Optional[float]) -> None:
    finished_at = get_datetime()
    started_at = finished_at - timedelta(seconds=duration)
    with pytest.raises(NegativeDurationError):
        create_past_job(dataset=DATASET, started_at=started_at, finished_at=finished_at)
