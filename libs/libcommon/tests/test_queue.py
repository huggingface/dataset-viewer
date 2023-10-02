# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
import os
import random
import time
from datetime import datetime, timedelta
from multiprocessing import Pool
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import pytest
import pytz

from libcommon.constants import QUEUE_TTL_SECONDS
from libcommon.queue import (
    EmptyQueueError,
    JobDocument,
    JobTotalMetricDocument,
    Lock,
    Queue,
    lock,
)
from libcommon.resources import QueueMongoResource
from libcommon.utils import Priority, Status, get_datetime

from .utils import assert_metric


def get_old_datetime() -> datetime:
    # Beware: the TTL index is set to 10 minutes. So it will delete the finished jobs after 10 minutes.
    # We have to use a datetime that is not older than 10 minutes.
    return get_datetime() - timedelta(seconds=(QUEUE_TTL_SECONDS / 2))


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


def test_add_job() -> None:
    test_type = "test_type"
    test_dataset = "test_dataset"
    test_revision = "test_revision"
    test_difficulty = 50
    # get the queue
    queue = Queue()
    assert JobTotalMetricDocument.objects().count() == 0
    # add a job
    job1 = queue.add_job(job_type=test_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    assert_metric(job_type=test_type, status=Status.WAITING, total=1)

    # a second call adds a second waiting job
    job2 = queue.add_job(job_type=test_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset, revision=test_revision)
    assert_metric(job_type=test_type, status=Status.WAITING, total=2)

    # get and start a job the second one should have been picked
    job_info = queue.start_job()
    assert job2.reload().status == Status.STARTED
    assert job_info["type"] == test_type
    assert job_info["params"]["dataset"] == test_dataset
    assert job_info["params"]["revision"] == test_revision
    assert job_info["params"]["config"] is None
    assert job_info["params"]["split"] is None
    assert_metric(job_type=test_type, status=Status.WAITING, total=1)
    assert_metric(job_type=test_type, status=Status.STARTED, total=1)

    # and the first job should have been cancelled
    assert job1.reload().status == Status.CANCELLED
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset, revision=test_revision)
    # adding the job while the first one has not finished yet adds another waiting job
    # (there are no limits to the number of waiting jobs)
    job3 = queue.add_job(job_type=test_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    assert job3.status == Status.WAITING
    assert_metric(job_type=test_type, status=Status.WAITING, total=2)
    assert_metric(job_type=test_type, status=Status.STARTED, total=1)

    with pytest.raises(EmptyQueueError):
        # but: it's not possible to start two jobs with the same arguments
        queue.start_job()
    # finish the first job
    queue.finish_job(job_id=job_info["job_id"], is_success=True)
    # the queue is not empty
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset, revision=test_revision)
    assert_metric(job_type=test_type, status=Status.WAITING, total=2)
    assert_metric(job_type=test_type, status=Status.STARTED, total=0)
    assert_metric(job_type=test_type, status=Status.SUCCESS, total=1)

    # process the third job
    job_info = queue.start_job()
    other_job_id = ("1" if job_info["job_id"][0] == "0" else "0") + job_info["job_id"][1:]
    assert_metric(job_type=test_type, status=Status.WAITING, total=1)
    assert_metric(job_type=test_type, status=Status.STARTED, total=1)
    assert_metric(job_type=test_type, status=Status.SUCCESS, total=1)

    # trying to finish another job fails silently (with a log)
    queue.finish_job(job_id=other_job_id, is_success=True)
    assert_metric(job_type=test_type, status=Status.WAITING, total=1)
    assert_metric(job_type=test_type, status=Status.STARTED, total=1)
    assert_metric(job_type=test_type, status=Status.SUCCESS, total=1)

    # finish it
    queue.finish_job(job_id=job_info["job_id"], is_success=True)
    assert_metric(job_type=test_type, status=Status.WAITING, total=1)
    assert_metric(job_type=test_type, status=Status.STARTED, total=0)
    assert_metric(job_type=test_type, status=Status.SUCCESS, total=2)

    # the queue is empty
    assert not queue.is_job_in_process(job_type=test_type, dataset=test_dataset, revision=test_revision)
    with pytest.raises(EmptyQueueError):
        # an error is raised if we try to start a job
        queue.start_job()


@pytest.mark.parametrize(
    "jobs_ids,job_ids_to_cancel,expected_canceled_number",
    [
        (["a", "b"], ["a", "b"], 2),
        (["a", "b"], ["a"], 1),
        (["a"], ["a", "b"], 1),
    ],
)
def test_cancel_jobs_by_job_id(
    jobs_ids: list[str], job_ids_to_cancel: list[str], expected_canceled_number: int
) -> None:
    test_type = "test_type"
    test_difficulty = 50
    queue = Queue()

    # we cannot really set job_id, so, we create jobs and get their job id, using dataset as a proxy
    real_job_ids_to_cancel = []
    waiting_jobs = 0
    for job_id in list(set(jobs_ids + job_ids_to_cancel)):
        job = queue.add_job(job_type=test_type, dataset=job_id, revision="test_revision", difficulty=test_difficulty)
        waiting_jobs += 1
        assert_metric(job_type=test_type, status=Status.WAITING, total=waiting_jobs)
        if job_id in job_ids_to_cancel:
            real_job_id = job.info()["job_id"]
            real_job_ids_to_cancel.append(real_job_id)
        if job_id not in jobs_ids:
            # delete the job, in order to simulate that it did never exist (we just wanted a valid job_id)
            job.delete()

    queue.start_job()
    assert_metric(job_type=test_type, status=Status.WAITING, total=1)
    assert_metric(job_type=test_type, status=Status.STARTED, total=1)
    canceled_number = queue.cancel_jobs_by_job_id(job_ids=real_job_ids_to_cancel)
    assert canceled_number == expected_canceled_number
    assert_metric(job_type=test_type, status=Status.CANCELLED, total=expected_canceled_number)


def test_cancel_jobs_by_job_id_wrong_format() -> None:
    queue = Queue()

    assert queue.cancel_jobs_by_job_id(job_ids=["not_a_valid_job_id"]) == 0
    assert JobTotalMetricDocument.objects().count() == 0


def check_job(queue: Queue, expected_dataset: str, expected_split: str, expected_priority: Priority) -> None:
    job_info = queue.start_job()
    assert job_info["params"]["dataset"] == expected_dataset
    assert job_info["params"]["split"] == expected_split
    assert job_info["priority"] == expected_priority


def test_priority_logic_creation_order() -> None:
    test_type = "test_type"
    test_revision = "test_revision"
    test_difficulty = 50
    queue = Queue()
    queue.add_job(
        job_type=test_type,
        dataset="dataset1",
        revision=test_revision,
        config="config",
        split="split1",
        difficulty=test_difficulty,
    )
    queue.add_job(
        job_type=test_type,
        dataset="dataset1",
        revision=test_revision,
        config="config",
        split="split2",
        difficulty=test_difficulty,
    )
    check_job(queue=queue, expected_dataset="dataset1", expected_split="split1", expected_priority=Priority.LOW)
    check_job(queue=queue, expected_dataset="dataset1", expected_split="split2", expected_priority=Priority.LOW)
    with pytest.raises(EmptyQueueError):
        queue.start_job()


def test_priority_logic_started_jobs_per_dataset_order() -> None:
    test_type = "test_type"
    test_revision = "test_revision"
    test_difficulty = 50
    queue = Queue()
    queue.add_job(
        job_type=test_type,
        dataset="dataset1",
        revision=test_revision,
        config="config",
        split="split1",
        difficulty=test_difficulty,
    )
    queue.add_job(
        job_type=test_type,
        dataset="dataset1",
        revision=test_revision,
        config="config",
        split="split2",
        difficulty=test_difficulty,
    )
    queue.add_job(
        job_type=test_type,
        dataset="dataset2",
        revision=test_revision,
        config="config",
        split="split1",
        difficulty=test_difficulty,
    )
    check_job(queue=queue, expected_dataset="dataset1", expected_split="split1", expected_priority=Priority.LOW)
    check_job(queue=queue, expected_dataset="dataset2", expected_split="split1", expected_priority=Priority.LOW)
    # ^ before, even if the creation date is after, because the dataset is different and has no started job
    check_job(queue=queue, expected_dataset="dataset1", expected_split="split2", expected_priority=Priority.LOW)
    with pytest.raises(EmptyQueueError):
        queue.start_job()


def test_priority_logic_started_jobs_per_namespace_order() -> None:
    test_type = "test_type"
    test_revision = "test_revision"
    test_difficulty = 50
    queue = Queue()
    queue.add_job(
        job_type=test_type,
        dataset="org1/dataset1",
        revision=test_revision,
        config="config",
        split="split1",
        difficulty=test_difficulty,
    )
    queue.add_job(
        job_type=test_type,
        dataset="org1/dataset2",
        revision=test_revision,
        config="config",
        split="split1",
        difficulty=test_difficulty,
    )
    queue.add_job(
        job_type=test_type,
        dataset="org2/dataset2",
        revision=test_revision,
        config="config",
        split="split1",
        difficulty=test_difficulty,
    )
    queue.add_job(
        job_type=test_type,
        dataset="no_org_dataset3",
        revision=test_revision,
        config="config",
        split="split1",
        difficulty=test_difficulty,
    )
    check_job(queue=queue, expected_dataset="org1/dataset1", expected_split="split1", expected_priority=Priority.LOW)
    check_job(queue=queue, expected_dataset="org2/dataset2", expected_split="split1", expected_priority=Priority.LOW)
    # ^ before, even if the creation date is after, because the namespace is different and has no started job
    check_job(queue=queue, expected_dataset="no_org_dataset3", expected_split="split1", expected_priority=Priority.LOW)
    check_job(queue=queue, expected_dataset="org1/dataset2", expected_split="split1", expected_priority=Priority.LOW)
    with pytest.raises(EmptyQueueError):
        queue.start_job()


def test_priority_logic_priority_order() -> None:
    test_type = "test_type"
    test_revision = "test_revision"
    test_difficulty = 50
    queue = Queue()
    queue.add_job(
        job_type=test_type,
        dataset="dataset1",
        revision=test_revision,
        config="config",
        split="split1",
        difficulty=test_difficulty,
    )
    queue.add_job(
        job_type=test_type,
        dataset="dataset2",
        revision=test_revision,
        config="config",
        split="split1",
        priority=Priority.NORMAL,
        difficulty=test_difficulty,
    )
    queue.add_job(
        job_type=test_type,
        dataset="dataset3",
        revision=test_revision,
        config="config",
        split="split1",
        priority=Priority.HIGH,
        difficulty=test_difficulty,
    )
    check_job(queue=queue, expected_dataset="dataset3", expected_split="split1", expected_priority=Priority.HIGH)
    # ^ before, even if the creation date is after, because the priority is higher
    check_job(queue=queue, expected_dataset="dataset2", expected_split="split1", expected_priority=Priority.NORMAL)
    # ^ before, even if the creation date is after, because the priority is higher
    check_job(queue=queue, expected_dataset="dataset1", expected_split="split1", expected_priority=Priority.LOW)
    with pytest.raises(EmptyQueueError):
        queue.start_job()


@pytest.mark.parametrize(
    "job_types_blocked,job_types_only,should_raise",
    [
        (None, None, False),
        (None, ["test_type"], False),
        (["other_type"], None, False),
        (["other_type"], ["test_type"], False),
        (None, ["other_type"], True),
        (["test_type"], None, True),
        (["test_type"], ["test_type"], True),
        (["other_type", "test_type"], None, True),
        (["other_type"], ["other_type"], True),
        (["other_type", "test_type"], ["other_type", "test_type"], True),
    ],
)
def test_job_types_only(
    job_types_blocked: Optional[list[str]], job_types_only: Optional[list[str]], should_raise: bool
) -> None:
    job_type = "test_type"
    test_dataset = "test_dataset"
    test_revision = "test_revision"
    test_difficulty = 50
    queue = Queue()
    queue.add_job(
        job_type=job_type,
        dataset=test_dataset,
        revision=test_revision,
        config=None,
        split=None,
        difficulty=test_difficulty,
    )
    assert queue.is_job_in_process(
        job_type=job_type, dataset=test_dataset, revision=test_revision, config=None, split=None
    )
    if should_raise:
        with pytest.raises(EmptyQueueError):
            queue.start_job(job_types_blocked=job_types_blocked, job_types_only=job_types_only)
    else:
        job_info = queue.start_job(job_types_blocked=job_types_blocked, job_types_only=job_types_only)
        assert job_info["params"]["dataset"] == test_dataset


@pytest.mark.parametrize(
    "difficulty_min,difficulty_max,should_raise",
    [
        (None, None, False),
        (None, 60, False),
        (40, None, False),
        (40, 60, False),
        (50, 50, False),
        (None, 40, True),
        (60, None, True),
        (60, 60, True),
        (40, 40, True),
        (55, 60, True),
        (40, 45, True),
    ],
)
def test_difficulty(difficulty_min: Optional[int], difficulty_max: Optional[int], should_raise: bool) -> None:
    job_type = "test_type"
    test_dataset = "test_dataset"
    test_revision = "test_revision"
    test_difficulty = 50
    queue = Queue()
    queue.add_job(
        job_type=job_type,
        dataset=test_dataset,
        revision=test_revision,
        config=None,
        split=None,
        difficulty=test_difficulty,
    )
    assert queue.is_job_in_process(
        job_type=job_type, dataset=test_dataset, revision=test_revision, config=None, split=None
    )
    if should_raise:
        with pytest.raises(EmptyQueueError):
            queue.start_job(difficulty_max=difficulty_max, difficulty_min=difficulty_min)
    else:
        job_info = queue.start_job(difficulty_max=difficulty_max, difficulty_min=difficulty_min)
        assert job_info["params"]["dataset"] == test_dataset


def test_count_by_status() -> None:
    test_type = "test_type"
    test_other_type = "test_other_type"
    test_dataset = "test_dataset"
    test_revision = "test_revision"
    test_difficulty = 50
    queue = Queue()

    expected_empty = {"waiting": 0, "started": 0, "success": 0, "error": 0, "cancelled": 0}
    expected_one_waiting = {"waiting": 1, "started": 0, "success": 0, "error": 0, "cancelled": 0}

    assert queue.get_jobs_count_by_status(job_type=test_type) == expected_empty
    assert queue.get_jobs_count_by_status(job_type=test_other_type) == expected_empty

    queue.add_job(job_type=test_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)

    assert queue.get_jobs_count_by_status(job_type=test_type) == expected_one_waiting
    assert queue.get_jobs_count_by_status(job_type=test_other_type) == expected_empty

    queue.add_job(job_type=test_other_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)

    assert queue.get_jobs_count_by_status(job_type=test_type) == expected_one_waiting
    assert queue.get_jobs_count_by_status(job_type=test_other_type) == expected_one_waiting


def test_get_dataset_pending_jobs_for_type() -> None:
    queue = Queue()
    test_type = "test_type"
    test_difficulty = 50
    test_another_type = "test_another_type"
    test_dataset = "test_dataset"
    test_another_dataset = "test_another_dataset"
    test_revision = "test_revision"
    test_configs_waiting = ["test_config_waiting_1", "test_config_waiting_2"]
    test_configs_started = ["test_config_started_1", "test_config_started_2"]
    test_configs_finished = ["test_config_finished_1", "test_config_finished_2"]
    for config in test_configs_finished:
        for dataset in [test_dataset, test_another_dataset]:
            for job_type in [test_type, test_another_type]:
                queue.add_job(
                    job_type=job_type,
                    dataset=dataset,
                    revision=test_revision,
                    config=config,
                    split=None,
                    difficulty=test_difficulty,
                )
                job_info = queue.start_job()
                queue.finish_job(job_info["job_id"], is_success=True)
    for config in test_configs_started:
        for dataset in [test_dataset, test_another_dataset]:
            for job_type in [test_type, test_another_type]:
                queue.add_job(
                    job_type=job_type,
                    dataset=dataset,
                    revision=test_revision,
                    config=config,
                    split=None,
                    difficulty=test_difficulty,
                )
                job_info = queue.start_job()
    for config in test_configs_waiting:
        for dataset in [test_dataset, test_another_dataset]:
            for job_type in [test_type, test_another_type]:
                queue.add_job(
                    job_type=job_type,
                    dataset=dataset,
                    revision=test_revision,
                    config=config,
                    split=None,
                    difficulty=test_difficulty,
                )
    result = queue.get_dataset_pending_jobs_for_type(dataset=test_dataset, job_type=test_type)
    assert len(result) == len(test_configs_waiting) + len(test_configs_started)
    for r in result:
        assert r["dataset"] == test_dataset
        assert r["type"] == test_type
        assert r["status"] in [Status.WAITING.value, Status.STARTED.value]


def test_queue_heartbeat() -> None:
    job_type = "test_type"
    test_difficulty = 50
    queue = Queue()
    job = queue.add_job(
        job_type=job_type,
        dataset="dataset1",
        revision="revision",
        config="config",
        split="split1",
        difficulty=test_difficulty,
    )
    queue.start_job(job_types_only=[job_type])
    assert job.last_heartbeat is None
    queue.heartbeat(job.pk)
    job.reload()
    assert job.last_heartbeat is not None
    last_heartbeat_datetime = pytz.UTC.localize(job.last_heartbeat)
    assert last_heartbeat_datetime >= get_datetime() - timedelta(seconds=1)


def test_queue_get_zombies() -> None:
    job_type = "test_type"
    test_difficulty = 50
    queue = Queue()
    with patch("libcommon.queue.get_datetime", get_old_datetime):
        zombie = queue.add_job(
            job_type=job_type,
            dataset="dataset1",
            revision="revision",
            config="config",
            split="split1",
            difficulty=test_difficulty,
        )
        queue.start_job(job_types_only=[job_type])
    queue.add_job(
        job_type=job_type,
        dataset="dataset1",
        revision="revision",
        config="config",
        split="split2",
        difficulty=test_difficulty,
    )
    queue.start_job(job_types_only=[job_type])
    assert queue.get_zombies(max_seconds_without_heartbeat=10) == [zombie.info()]
    assert queue.get_zombies(max_seconds_without_heartbeat=-1) == []
    assert queue.get_zombies(max_seconds_without_heartbeat=0) == []
    assert queue.get_zombies(max_seconds_without_heartbeat=9999999) == []


def test_has_ttl_index_on_finished_at_field() -> None:
    ttl_index_names = [
        name
        for name, value in JobDocument._get_collection().index_information().items()
        if "expireAfterSeconds" in value and "key" in value and value["key"] == [("finished_at", 1)]
    ]
    assert len(ttl_index_names) == 1
    ttl_index_name = ttl_index_names[0]
    assert ttl_index_name == "finished_at_1"
    assert JobDocument._get_collection().index_information()[ttl_index_name]["expireAfterSeconds"] == QUEUE_TTL_SECONDS


def random_sleep() -> None:
    MAX_SLEEP_MS = 40
    time.sleep(MAX_SLEEP_MS / 1000 * random.random())


def increment(tmp_file: Path) -> None:
    random_sleep()
    with open(tmp_file, "r") as f:
        current = int(f.read() or 0)
    random_sleep()
    with open(tmp_file, "w") as f:
        f.write(str(current + 1))
    random_sleep()


def locked_increment(tmp_file: Path) -> None:
    sleeps = [0.05, 0.05, 0.05, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5]
    with lock(key="test_lock", owner=str(os.getpid()), sleeps=sleeps):
        increment(tmp_file)


def test_lock(tmp_path_factory: pytest.TempPathFactory, queue_mongo_resource: QueueMongoResource) -> None:
    tmp_file = Path(tmp_path_factory.mktemp("test_lock") / "tmp.txt")
    tmp_file.touch()
    max_parallel_jobs = 4
    num_jobs = 42

    with Pool(max_parallel_jobs, initializer=queue_mongo_resource.allocate) as pool:
        pool.map(locked_increment, [tmp_file] * num_jobs)

    expected = num_jobs
    with open(tmp_file, "r") as f:
        assert int(f.read()) == expected
    Lock.objects(key="test_lock").delete()


def git_branch_locked_increment(tmp_file: Path) -> None:
    sleeps = [0.05, 0.05, 0.05, 1, 1, 1, 1, 1, 1, 5, 5, 5, 5]
    dataset = "dataset"
    branch = "refs/convert/parquet"
    with lock.git_branch(dataset=dataset, branch=branch, owner=str(os.getpid()), sleeps=sleeps):
        increment(tmp_file)


def test_lock_git_branch(tmp_path_factory: pytest.TempPathFactory, queue_mongo_resource: QueueMongoResource) -> None:
    tmp_file = Path(tmp_path_factory.mktemp("test_lock") / "tmp.txt")
    tmp_file.touch()
    max_parallel_jobs = 5
    num_jobs = 43

    with Pool(max_parallel_jobs, initializer=queue_mongo_resource.allocate) as pool:
        pool.map(git_branch_locked_increment, [tmp_file] * num_jobs)

    expected = num_jobs
    with open(tmp_file, "r") as f:
        assert int(f.read()) == expected
    assert Lock.objects().count() == 1
    assert Lock.objects().get().key == json.dumps({"dataset": "dataset", "branch": "refs/convert/parquet"})
    assert Lock.objects().get().owner is None
    Lock.objects().delete()


def test_cancel_dataset_jobs(queue_mongo_resource: QueueMongoResource) -> None:
    """
    Test that cancel_dataset_jobs cancels all jobs for a dataset

    -> cancels at several levels (dataset, config, split)
    -> cancels started and waiting jobs
    -> remove locks
    -> does not cancel, and does not remove locks, for other datasets
    """
    dataset = "dataset"
    other_dataset = "other_dataset"
    job_type_1 = "job_type_1"
    job_type_2 = "job_type_2"
    job_type_3 = "job_type_3"
    job_type_4 = "job_type_4"
    revision = "not important"
    difficulty = 50
    queue = Queue()
    queue.add_job(
        job_type=job_type_1,
        dataset=dataset,
        revision=revision,
        config=None,
        split=None,
        difficulty=difficulty,
    )
    queue.add_job(
        job_type=job_type_1,
        dataset=other_dataset,
        revision=revision,
        config=None,
        split=None,
        difficulty=difficulty,
    )
    queue.add_job(
        job_type=job_type_2,
        dataset=dataset,
        revision=revision,
        config=None,
        split=None,
        difficulty=difficulty,
    )
    queue.add_job(
        job_type=job_type_3,
        dataset=dataset,
        revision=revision,
        config="config",
        split=None,
        difficulty=difficulty,
    )
    queue.add_job(
        job_type=job_type_4,
        dataset=dataset,
        revision=revision,
        config="config",
        split="split",
        difficulty=difficulty,
    )
    queue.add_job(
        job_type=job_type_2,
        dataset=other_dataset,
        revision=revision,
        config=None,
        split=None,
        difficulty=difficulty,
    )
    started_job_info_1 = queue.start_job()
    assert started_job_info_1["params"]["dataset"] == dataset
    assert started_job_info_1["type"] == job_type_1
    started_job_info_2 = queue.start_job()
    assert started_job_info_2["params"]["dataset"] == other_dataset
    assert started_job_info_2["type"] == job_type_1

    queue.cancel_dataset_jobs(dataset=dataset)

    assert JobDocument.objects().count() == 6
    assert JobDocument.objects(dataset=dataset).count() == 4
    assert JobDocument.objects(dataset=dataset, status=Status.CANCELLED).count() == 4
    assert JobDocument.objects(dataset=other_dataset).count() == 2
    assert JobDocument.objects(dataset=other_dataset, status=Status.STARTED).count() == 1
    assert JobDocument.objects(dataset=other_dataset, status=Status.WAITING).count() == 1

    assert len(Lock.objects()) == 2
    assert len(Lock.objects(key=f"{job_type_1},{dataset}", owner=None)) == 1
    assert len(Lock.objects(key=f"{job_type_1},{dataset}", owner__ne=None)) == 0
    # ^ does not test much, because at that time, the lock should already have been released
