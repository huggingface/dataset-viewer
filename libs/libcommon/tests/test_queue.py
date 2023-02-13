# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import time
from typing import Iterator, Optional

import pytest

from libcommon.queue import (
    EmptyQueueError,
    Priority,
    Queue,
    Status,
    _clean_queue_database,
)
from libcommon.resources import QueueMongoResource


@pytest.fixture(autouse=True)
def queue_mongo_resource(queue_mongo_host: str) -> Iterator[QueueMongoResource]:
    database = "datasets_server_queue_test"
    host = queue_mongo_host
    if "test" not in database:
        raise ValueError("Test must be launched on a test mongo database")
    with QueueMongoResource(database=database, host=host, server_selection_timeout_ms=3_000) as queue_mongo_resource:
        if not queue_mongo_resource.is_available():
            raise RuntimeError("Mongo resource is not available")
        yield queue_mongo_resource
        _clean_queue_database()


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
    assert job_info["dataset"] == test_dataset
    assert job_info["config"] is None
    assert job_info["split"] is None
    assert job_info["force"] is True
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
    assert job_info["force"] is False
    queue.finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)
    # and the third one
    job_info = queue.start_job()
    assert job_info["force"] is True
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
    assert job_info["dataset"] == test_dataset
    assert job_info["config"] is None
    assert job_info["split"] is None
    assert job_info["force"] is True  # the new job inherits from waiting forced jobs
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
    assert job_info["force"] is False  # the new jobs does not inherit from started forced jobs
    queue.finish_job(job_id=job_info["job_id"], finished_status=Status.SUCCESS)
    # the queue is empty
    assert not queue.is_job_in_process(job_type=test_type, dataset=test_dataset)
    with pytest.raises(EmptyQueueError):
        # an error is raised if we try to start a job
        queue.start_job()


def check_job(queue: Queue, expected_dataset: str, expected_split: str) -> None:
    job_info = queue.start_job()
    assert job_info["dataset"] == expected_dataset
    assert job_info["split"] == expected_split


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
    assert (
        queue.is_job_in_process(job_type=test_type, dataset=test_dataset, config=test_config, split="split1") is True
    )
    queue.upsert_job(job_type=test_type, dataset=test_dataset, config=test_config, split="split2")
    queue.upsert_job(job_type=test_type, dataset=test_dataset, config=test_config, split="split3")
    job_info = queue.start_job()
    assert job_info["dataset"] == test_dataset
    assert job_info["config"] == test_config
    assert job_info["split"] == "split1"
    assert (
        queue.is_job_in_process(job_type=test_type, dataset=test_dataset, config=test_config, split="split1")
    )
    if max_jobs_per_namespace == 1:
        with pytest.raises(EmptyQueueError):
            queue.start_job()
        return
    job_info_2 = queue.start_job()
    assert job_info_2["split"] == "split2"
    if max_jobs_per_namespace == 2:
        with pytest.raises(EmptyQueueError):
            queue.start_job()
        return
    # max_jobs_per_namespace <= 0 and max_jobs_per_namespace == None are the same
    # finish the first job
    queue.finish_job(job_info["job_id"], finished_status=Status.SUCCESS)
    assert (
        queue.is_job_in_process(job_type=test_type, dataset=test_dataset, config=test_config, split="split1") is False
    )


@pytest.mark.parametrize(
    "job_type,only_job_types",
    [
        ("test_type", None),
        ("test_type", ["test_type"]),
        ("test_type", ["other_type", "test_type"]),
        ("test_type", ["other_type"]),
    ],
)
def test_only_job_types(job_type: str, only_job_types: Optional[list[str]]) -> None:
    test_dataset = "test_dataset"
    queue = Queue(max_jobs_per_namespace=100)
    queue.upsert_job(job_type=job_type, dataset=test_dataset, config=None, split=None)
    assert queue.is_job_in_process(job_type=job_type, dataset=test_dataset, config=None, split=None)
    if only_job_types and job_type not in only_job_types:
        with pytest.raises(EmptyQueueError):
            job_info = queue.start_job(only_job_types=only_job_types)
            queue.start_job(only_job_types=only_job_types)
    else:
        job_info = queue.start_job(only_job_types=only_job_types)
        assert job_info["dataset"] == test_dataset


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


def test_get_total_duration_per_dataset() -> None:
    test_type = "test_type"
    test_dataset = "test_dataset"
    test_config = "test_config"
    queue = Queue()
    queue.upsert_job(job_type=test_type, dataset=test_dataset, config=test_config, split="split1")
    queue.upsert_job(job_type=test_type, dataset=test_dataset, config=test_config, split="split2")
    queue.upsert_job(job_type=test_type, dataset=test_dataset, config=test_config, split="split3")
    queue.upsert_job(job_type=test_type, dataset=test_dataset, config=test_config, split="split4")
    queue.upsert_job(job_type=test_type, dataset=test_dataset, config=test_config, split="split5")
    job_info = queue.start_job()
    job_info_2 = queue.start_job()
    job_info_3 = queue.start_job()
    _ = queue.start_job()
    duration = 2
    time.sleep(duration)
    # finish three jobs
    queue.finish_job(job_info["job_id"], finished_status=Status.SUCCESS)
    queue.finish_job(job_info_2["job_id"], finished_status=Status.ERROR)
    queue.finish_job(job_info_3["job_id"], finished_status=Status.SUCCESS)
    # cancel one remaining job
    queue.cancel_started_jobs(job_type=test_type)
    # check the total duration
    assert queue.get_total_duration_per_dataset(job_type=test_type)[test_dataset] >= duration * 3
    # ^ it should be equal,  not >=, but if the runner is slow, it might take a bit more time
