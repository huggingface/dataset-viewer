# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from datetime import datetime, timedelta
from typing import List, Optional
from unittest.mock import patch

import pytest
import pytz

from libcommon.constants import QUEUE_TTL_SECONDS
from libcommon.queue import EmptyQueueError, Job, Queue
from libcommon.resources import QueueMongoResource
from libcommon.utils import Priority, Status, get_datetime


def get_old_datetime() -> datetime:
    return get_datetime() - timedelta(days=1)


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


def test__add_job() -> None:
    test_type = "test_type"
    test_dataset = "test_dataset"
    # get the queue
    queue = Queue()
    # add a job
    queue._add_job(job_type=test_type, dataset=test_dataset, force=True)
    # a second call adds a second waiting job
    queue._add_job(job_type=test_type, dataset=test_dataset)
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset)
    # get and start the first job
    job_info = queue.start_job()
    assert job_info["type"] == test_type
    assert job_info["params"]["dataset"] == test_dataset
    assert job_info["params"]["config"] is None
    assert job_info["params"]["split"] is None
    assert job_info["force"]
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset)
    # adding the job while the first one has not finished yet adds another waiting job
    # (there are no limits to the number of waiting jobs)
    queue._add_job(job_type=test_type, dataset=test_dataset, force=True)
    with pytest.raises(EmptyQueueError):
        # but: it's not possible to start two jobs with the same arguments
        queue.start_job()
    # finish the first job
    queue.finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)
    # the queue is not empty
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset)
    # process the second job
    job_info = queue.start_job()
    assert not job_info["force"]
    queue.finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)
    # and the third one
    job_info = queue.start_job()
    assert job_info["force"]
    other_job_id = ("1" if job_info["job_id"][0] == "0" else "0") + job_info["job_id"][1:]
    # trying to finish another job fails silently (with a log)
    queue.finish_job(job_id=other_job_id, finished_status=Status.SUCCESS)
    # finish it
    queue.finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)
    # the queue is empty
    assert not queue.is_job_in_process(job_type=test_type, dataset=test_dataset)
    with pytest.raises(EmptyQueueError):
        # an error is raised if we try to start a job
        queue.start_job()


def test_upsert_job() -> None:
    test_type = "test_type"
    test_dataset = "test_dataset"
    # get the queue
    queue = Queue()
    # upsert a job
    queue.upsert_job(job_type=test_type, dataset=test_dataset, force=True)
    # a second call creates a second waiting job, and the first one is cancelled
    queue.upsert_job(job_type=test_type, dataset=test_dataset)
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset)
    # get and start the last job
    job_info = queue.start_job()
    assert job_info["type"] == test_type
    assert job_info["params"]["dataset"] == test_dataset
    assert job_info["params"]["config"] is None
    assert job_info["params"]["split"] is None
    assert job_info["force"]  # the new job inherits from waiting forced jobs
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset)
    # adding the job while the first one has not finished yet adds a new waiting job
    queue.upsert_job(job_type=test_type, dataset=test_dataset, force=False)
    with pytest.raises(EmptyQueueError):
        # but: it's not possible to start two jobs with the same arguments
        queue.start_job()
    # finish the first job
    queue.finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)
    # the queue is not empty
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset)
    # process the second job
    job_info = queue.start_job()
    assert not job_info["force"]  # the new jobs does not inherit from started forced jobs
    queue.finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)
    # the queue is empty
    assert not queue.is_job_in_process(job_type=test_type, dataset=test_dataset)
    with pytest.raises(EmptyQueueError):
        # an error is raised if we try to start a job
        queue.start_job()


@pytest.mark.parametrize(
    "statuses_to_cancel, expected_remaining_number",
    [
        (None, 0),
        ([Status.WAITING], 1),
        ([Status.WAITING, Status.STARTED], 0),
        ([Status.STARTED], 1),
        ([Status.SUCCESS], 2),
    ],
)
def test_cancel_jobs(statuses_to_cancel: Optional[List[Status]], expected_remaining_number: int) -> None:
    test_type = "test_type"
    test_dataset = "test_dataset"
    queue = Queue()
    queue._add_job(job_type=test_type, dataset=test_dataset, force=True)
    queue._add_job(job_type=test_type, dataset=test_dataset)
    queue.start_job()

    canceled_job_dicts = queue.cancel_jobs(
        job_type=test_type, dataset=test_dataset, statuses_to_cancel=statuses_to_cancel
    )
    assert len(canceled_job_dicts) == 2 - expected_remaining_number

    if expected_remaining_number == 0:
        assert not queue.is_job_in_process(job_type=test_type, dataset=test_dataset)
        with pytest.raises(EmptyQueueError):
            queue.start_job()
    else:
        assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset)


def check_job(queue: Queue, expected_dataset: str, expected_split: str) -> None:
    job_info = queue.start_job()
    assert job_info["params"]["dataset"] == expected_dataset
    assert job_info["params"]["split"] == expected_split


def test_priority_logic() -> None:
    test_type = "test_type"
    queue = Queue()
    queue.upsert_job(job_type=test_type, dataset="dataset1", config="config", split="split1")
    queue.upsert_job(job_type=test_type, dataset="dataset1/dataset", config="config", split="split1")
    queue.upsert_job(job_type=test_type, dataset="dataset1", config="config", split="split2")
    queue.upsert_job(job_type=test_type, dataset="dataset2", config="config", split="split1", priority=Priority.LOW)
    queue.upsert_job(
        job_type=test_type, dataset="dataset2/dataset", config="config", split="split1", priority=Priority.LOW
    )
    queue.upsert_job(job_type=test_type, dataset="dataset2", config="config", split="split2")
    queue.upsert_job(job_type=test_type, dataset="dataset3", config="config", split="split1")
    queue.upsert_job(job_type=test_type, dataset="dataset3", config="config", split="split1", priority=Priority.LOW)
    queue.upsert_job(job_type=test_type, dataset="dataset1", config="config", split="split1")
    queue.upsert_job(job_type=test_type, dataset="dataset2", config="config", split="split1", priority=Priority.LOW)
    check_job(queue=queue, expected_dataset="dataset1/dataset", expected_split="split1")
    check_job(queue=queue, expected_dataset="dataset2", expected_split="split2")
    check_job(queue=queue, expected_dataset="dataset3", expected_split="split1")
    # ^ before the other "dataset3" jobs because its priority is higher (it inherited Priority.NORMAL in upsert_job)
    check_job(queue=queue, expected_dataset="dataset1", expected_split="split2")
    # ^ same namespace as dataset1/dataset, goes after namespaces without any started job
    check_job(queue=queue, expected_dataset="dataset1", expected_split="split1")
    # ^ comes after the other "dataset1" jobs because the last upsert_job call moved its creation date
    check_job(queue=queue, expected_dataset="dataset2/dataset", expected_split="split1")
    # ^ comes after the other "dataset2" jobs because its priority is lower
    check_job(queue=queue, expected_dataset="dataset2", expected_split="split1")
    # ^ the rest of the rules apply for Priority.LOW jobs
    with pytest.raises(EmptyQueueError):
        queue.start_job()


@pytest.mark.parametrize("max_jobs_per_namespace", [(None), (-5), (0), (1), (2)])
def test_max_jobs_per_namespace(max_jobs_per_namespace: Optional[int]) -> None:
    test_type = "test_type"
    test_dataset = "test_dataset"
    test_config = "test_config"
    queue = Queue(max_jobs_per_namespace=max_jobs_per_namespace)
    queue.upsert_job(job_type=test_type, dataset=test_dataset, config=test_config, split="split1")
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset, config=test_config, split="split1")
    queue.upsert_job(job_type=test_type, dataset=test_dataset, config=test_config, split="split2")
    queue.upsert_job(job_type=test_type, dataset=test_dataset, config=test_config, split="split3")
    job_info = queue.start_job()
    assert job_info["params"]["dataset"] == test_dataset
    assert job_info["params"]["config"] == test_config
    assert job_info["params"]["split"] == "split1"
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset, config=test_config, split="split1")
    if max_jobs_per_namespace == 1:
        with pytest.raises(EmptyQueueError):
            queue.start_job()
        return
    job_info_2 = queue.start_job()
    assert job_info_2["params"]["split"] == "split2"
    if max_jobs_per_namespace == 2:
        with pytest.raises(EmptyQueueError):
            queue.start_job()
        return
    # max_jobs_per_namespace <= 0 and max_jobs_per_namespace == None are the same
    # finish the first job
    queue.finish_job(job_info["job_id"], finished_status=Status.SUCCESS)
    assert not queue.is_job_in_process(job_type=test_type, dataset=test_dataset, config=test_config, split="split1")


@pytest.mark.parametrize(
    "job_type,job_types_blocked,job_types_only,should_raise",
    [
        ("test_type", None, None, False),
        ("test_type", None, ["test_type"], False),
        ("test_type", ["other_type"], None, False),
        ("test_type", ["other_type"], ["test_type"], False),
        ("test_type", None, ["other_type"], True),
        ("test_type", ["test_type"], None, True),
        ("test_type", ["test_type"], ["test_type"], True),
        ("test_type", ["other_type", "test_type"], None, True),
        ("test_type", ["other_type"], ["other_type"], True),
        ("test_type", ["other_type", "test_type"], ["other_type", "test_type"], True),
    ],
)
def test_job_types_only(
    job_type: str, job_types_blocked: Optional[list[str]], job_types_only: Optional[list[str]], should_raise: bool
) -> None:
    test_dataset = "test_dataset"
    queue = Queue(max_jobs_per_namespace=100)
    queue.upsert_job(job_type=job_type, dataset=test_dataset, config=None, split=None)
    assert queue.is_job_in_process(job_type=job_type, dataset=test_dataset, config=None, split=None)
    if should_raise:
        with pytest.raises(EmptyQueueError):
            queue.start_job(job_types_blocked=job_types_blocked, job_types_only=job_types_only)
    else:
        job_info = queue.start_job(job_types_blocked=job_types_blocked, job_types_only=job_types_only)
        assert job_info["params"]["dataset"] == test_dataset


def test_count_by_status() -> None:
    test_type = "test_type"
    test_other_type = "test_other_type"
    test_dataset = "test_dataset"
    queue = Queue()

    expected_empty = {"waiting": 0, "started": 0, "success": 0, "error": 0, "cancelled": 0, "skipped": 0}
    expected_one_waiting = {"waiting": 1, "started": 0, "success": 0, "error": 0, "cancelled": 0, "skipped": 0}

    assert queue.get_jobs_count_by_status(job_type=test_type) == expected_empty
    assert queue.get_jobs_count_by_status(job_type=test_other_type) == expected_empty

    queue.upsert_job(job_type=test_type, dataset=test_dataset)

    assert queue.get_jobs_count_by_status(job_type=test_type) == expected_one_waiting
    assert queue.get_jobs_count_by_status(job_type=test_other_type) == expected_empty

    queue.upsert_job(job_type=test_other_type, dataset=test_dataset)

    assert queue.get_jobs_count_by_status(job_type=test_type) == expected_one_waiting
    assert queue.get_jobs_count_by_status(job_type=test_other_type) == expected_one_waiting


def test_get_dataset_pending_jobs_for_type() -> None:
    queue = Queue(max_jobs_per_namespace=100)
    test_type = "test_type"
    test_another_type = "test_another_type"
    test_dataset = "test_dataset"
    test_another_dataset = "test_another_dataset"
    test_configs_waiting = ["test_config_waiting_1", "test_config_waiting_2"]
    test_configs_started = ["test_config_started_1", "test_config_started_2"]
    test_configs_finished = ["test_config_finished_1", "test_config_finished_2"]
    for config in test_configs_finished:
        for dataset in [test_dataset, test_another_dataset]:
            for job_type in [test_type, test_another_type]:
                queue.upsert_job(job_type=job_type, dataset=dataset, config=config, split=None)
                job_info = queue.start_job()
                queue.finish_job(job_info["job_id"], finished_status=Status.SUCCESS)
    for config in test_configs_started:
        for dataset in [test_dataset, test_another_dataset]:
            for job_type in [test_type, test_another_type]:
                queue.upsert_job(job_type=job_type, dataset=dataset, config=config, split=None)
                job_info = queue.start_job()
    for config in test_configs_waiting:
        for dataset in [test_dataset, test_another_dataset]:
            for job_type in [test_type, test_another_type]:
                queue.upsert_job(job_type=job_type, dataset=dataset, config=config, split=None)
    result = queue.get_dataset_pending_jobs_for_type(dataset=test_dataset, job_type=test_type)
    assert len(result) == len(test_configs_waiting) + len(test_configs_started)
    for r in result:
        assert r["dataset"] == test_dataset
        assert r["type"] == test_type
        assert r["status"] in [Status.WAITING.value, Status.STARTED.value]


def test_queue_heartbeat() -> None:
    job_type = "test_type"
    queue = Queue()
    job = queue.upsert_job(job_type=job_type, dataset="dataset1", config="config", split="split1")
    queue.start_job(job_types_only=[job_type])
    assert job.last_heartbeat is None
    queue.heartbeat(job.pk)
    job.reload()
    assert job.last_heartbeat is not None
    last_heartbeat_datetime = pytz.UTC.localize(job.last_heartbeat)
    assert last_heartbeat_datetime >= get_datetime() - timedelta(seconds=1)


def test_queue_get_zombies() -> None:
    job_type = "test_type"
    queue = Queue()
    with patch("libcommon.queue.get_datetime", get_old_datetime):
        zombie = queue.upsert_job(job_type=job_type, dataset="dataset1", config="config", split="split1")
        queue.start_job(job_types_only=[job_type])
    queue.upsert_job(job_type=job_type, dataset="dataset1", config="config", split="split2")
    queue.start_job(job_types_only=[job_type])
    assert queue.get_zombies(max_seconds_without_heartbeat=10) == [zombie.info()]
    assert queue.get_zombies(max_seconds_without_heartbeat=-1) == []
    assert queue.get_zombies(max_seconds_without_heartbeat=0) == []
    assert queue.get_zombies(max_seconds_without_heartbeat=9999999) == []


def test_queue_kill_zombies() -> None:
    job_type = "test_type"
    queue = Queue()
    with patch("libcommon.queue.get_datetime", get_old_datetime):
        zombie = queue.upsert_job(job_type=job_type, dataset="dataset1", config="config", split="split1")
        queue.start_job(job_types_only=[job_type])
    another_job = queue.upsert_job(job_type=job_type, dataset="dataset1", config="config", split="split2")
    queue.start_job(job_types_only=[job_type])

    assert queue.get_zombies(max_seconds_without_heartbeat=10) == [zombie.info()]
    queue.kill_zombies([zombie.info()])
    assert queue.get_zombies(max_seconds_without_heartbeat=10) == []
    zombie.reload()
    another_job.reload()
    assert zombie.status == Status.ERROR
    assert another_job.status == Status.STARTED


def test_has_ttl_index_on_finished_at_field() -> None:
    ttl_index_names = [
        name
        for name, value in Job._get_collection().index_information().items()
        if "expireAfterSeconds" in value and "key" in value and value["key"] == [("finished_at", 1)]
    ]
    assert len(ttl_index_names) == 1
    ttl_index_name = ttl_index_names[0]
    assert ttl_index_name == "finished_at_1"
    assert Job._get_collection().index_information()[ttl_index_name]["expireAfterSeconds"] == QUEUE_TTL_SECONDS
