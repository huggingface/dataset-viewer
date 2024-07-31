# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import patch

import pytest
import pytz

from libcommon.constants import QUEUE_TTL_SECONDS
from libcommon.dtos import Priority, Status, WorkerSize
from libcommon.queue.dataset_blockages import DATASET_STATUS_NORMAL, block_dataset
from libcommon.queue.jobs import EmptyQueueError, JobDocument, Queue
from libcommon.queue.metrics import JobTotalMetricDocument, WorkerSizeJobsCountDocument
from libcommon.queue.past_jobs import JOB_DURATION_MIN_SECONDS, PastJobDocument
from libcommon.resources import QueueMongoResource
from libcommon.utils import get_datetime


def assert_metric_jobs_per_type(
    job_type: str, status: str, total: int, dataset_status: str = DATASET_STATUS_NORMAL
) -> None:
    metric = JobTotalMetricDocument.objects(job_type=job_type, status=status, dataset_status=dataset_status).first()
    assert metric is not None
    assert metric.total == total


def assert_metric_jobs_per_worker(worker_size: str, jobs_count: int) -> None:
    metric = WorkerSizeJobsCountDocument.objects(worker_size=worker_size).first()
    assert metric is not None, metric
    assert metric.jobs_count == jobs_count, metric.jobs_count


def assert_past_jobs_number(count: int) -> None:
    assert PastJobDocument.objects().count() == count


def get_old_datetime() -> datetime:
    # Beware: the TTL index is set to 10 minutes. So it will delete the finished jobs after 10 minutes.
    # We have to use a datetime that is not older than 10 minutes.
    return get_datetime() - timedelta(seconds=(QUEUE_TTL_SECONDS / 2))


def get_future_datetime() -> datetime:
    return get_datetime() + timedelta(seconds=JOB_DURATION_MIN_SECONDS * 2)


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
    assert WorkerSizeJobsCountDocument.objects().count() == 0
    assert_past_jobs_number(0)

    # add a job
    job1 = queue.add_job(job_type=test_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=1)
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=1)

    # a second call adds a second waiting job
    job2 = queue.add_job(job_type=test_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset, revision=test_revision)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=2)
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=2)

    # get and start a job the second one should have been picked
    job_info = queue.start_job()
    assert job2.reload().status == Status.STARTED
    assert job_info["type"] == test_type
    assert job_info["params"]["dataset"] == test_dataset
    assert job_info["params"]["revision"] == test_revision
    assert job_info["params"]["config"] is None
    assert job_info["params"]["split"] is None
    # it should have deleted the other waiting jobs for the same unicity_id
    assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=0)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.STARTED, total=1)
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=0)

    # and the first job should have been deleted
    assert JobDocument.objects(pk=job1.pk).count() == 0
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset, revision=test_revision)
    # adding the job while the first one has not finished yet adds another waiting job
    # (there are no limits to the number of waiting jobs)
    job3 = queue.add_job(job_type=test_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    assert job3.status == Status.WAITING
    assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=1)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.STARTED, total=1)
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=1)

    with pytest.raises(EmptyQueueError):
        # but: it's not possible to start two jobs with the same arguments
        queue.start_job()
    # finish the first job
    assert_past_jobs_number(0)
    queue.finish_job(job_id=job_info["job_id"])
    assert_past_jobs_number(0)
    # ^ the duration is too short, it's ignored

    # the queue is not empty
    assert queue.is_job_in_process(job_type=test_type, dataset=test_dataset, revision=test_revision)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=1)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.STARTED, total=0)
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=1)

    # process the third job
    job_info = queue.start_job()
    other_job_id = ("1" if job_info["job_id"][0] == "0" else "0") + job_info["job_id"][1:]
    assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=0)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.STARTED, total=1)
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=0)

    # trying to finish another job fails silently (with a log)
    queue.finish_job(job_id=other_job_id)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=0)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.STARTED, total=1)
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=0)

    # finish it (but changing the start date, so that an entry is created in pastJobs)
    with patch("libcommon.queue.jobs.get_datetime", get_future_datetime):
        queue.finish_job(job_id=job_info["job_id"])
    assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=0)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.STARTED, total=0)
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=0)

    # the queue is empty
    assert not queue.is_job_in_process(job_type=test_type, dataset=test_dataset, revision=test_revision)
    with pytest.raises(EmptyQueueError):
        # an error is raised if we try to start a job
        queue.start_job()

    # one long finished job
    assert_past_jobs_number(1)


def test_finish_job_blocked() -> None:
    test_type = "test_type"
    test_dataset = "test_dataset"
    test_revision = "test_revision"
    test_difficulty = 50

    queue = Queue()
    assert WorkerSizeJobsCountDocument.objects().count() == 0

    queue.add_job(job_type=test_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    queue.add_job(job_type="test_type2", dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    queue.add_job(job_type="test_type3", dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    queue.add_job(job_type="test_type4", dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)

    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=4)

    job_info = queue.start_job()
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=3)

    with patch("libcommon.queue.jobs.create_past_job", return_value=True):
        queue.finish_job(job_id=job_info["job_id"])
        assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=0)


@pytest.mark.parametrize(
    "jobs_ids,job_ids_to_delete,expected_deleted_number",
    [
        (["a", "b"], ["a", "b"], 1),
        (["a", "b"], ["b"], 1),
        (["a"], ["a", "b"], 0),
    ],
)
def test_delete_waiting_jobs_by_job_id(
    jobs_ids: list[str], job_ids_to_delete: list[str], expected_deleted_number: int
) -> None:
    test_type = "test_type"
    test_difficulty = 50
    queue = Queue()

    # we cannot really set job_id, so, we create jobs and get their job id, using dataset as a proxy
    real_job_ids_to_delete = []
    waiting_jobs = 0
    all_jobs = sorted(list(set(jobs_ids + job_ids_to_delete)))  # ensure always 'a' is first started
    for job_id in all_jobs:
        job = queue.add_job(job_type=test_type, dataset=job_id, revision="test_revision", difficulty=test_difficulty)
        waiting_jobs += 1
        assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=waiting_jobs)
        assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=waiting_jobs)
        if job_id in job_ids_to_delete:
            real_job_id = job.info()["job_id"]
            real_job_ids_to_delete.append(real_job_id)
        if job_id not in jobs_ids:
            # delete the job, in order to simulate that it did never exist (we just wanted a valid job_id)
            job.delete()
    assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=len(all_jobs))
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=len(all_jobs))

    queue.start_job()
    assert_metric_jobs_per_type(job_type=test_type, status=Status.WAITING, total=len(all_jobs) - 1)
    assert_metric_jobs_per_type(job_type=test_type, status=Status.STARTED, total=1)
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=len(all_jobs) - 1)
    deleted_number = queue.delete_waiting_jobs_by_job_id(job_ids=real_job_ids_to_delete)
    assert deleted_number == expected_deleted_number


def test_delete_waiting_jobs_by_job_id_wrong_format() -> None:
    queue = Queue()

    assert queue.delete_waiting_jobs_by_job_id(job_ids=["not_a_valid_job_id"]) == 0
    assert JobTotalMetricDocument.objects().count() == 0


def test_delete_waiting_jobs_by_job_id_blocked() -> None:
    queue = Queue()
    job = queue.add_job(job_type="test_type", dataset="dataset", revision="test_revision", difficulty=50)
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=1)
    assert queue.delete_waiting_jobs_by_job_id(job_ids=[job.info()["job_id"]]) == 1
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=0)

    job = queue.add_job(job_type="test_type", dataset="dataset", revision="test_revision", difficulty=100)
    block_dataset("dataset")
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=0)
    assert queue.delete_waiting_jobs_by_job_id(job_ids=[job.info()["job_id"]]) == 1
    assert_metric_jobs_per_worker(worker_size=WorkerSize.medium, jobs_count=0)


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
    "difficulty_min,difficulty_max,should_raise",
    [
        (None, None, False),
        (None, 60, False),
        (40, None, False),
        (40, 60, False),
        (40, 50, False),
        (50, 50, True),
        (50, 60, True),
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


def test_get_jobs_total_by_type_status_and_dataset_status() -> None:
    test_type = "test_type"
    test_other_type = "test_other_type"
    test_dataset = "test_dataset"
    test_revision = "test_revision"
    test_difficulty = 50
    queue = Queue()

    assert queue.get_jobs_total_by_type_status_and_dataset_status() == {}
    queue.add_job(job_type=test_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    assert queue.get_jobs_total_by_type_status_and_dataset_status() == {
        (test_type, "waiting", DATASET_STATUS_NORMAL): 1
    }
    queue.add_job(job_type=test_other_type, dataset=test_dataset, revision=test_revision, difficulty=test_difficulty)
    assert queue.get_jobs_total_by_type_status_and_dataset_status() == {
        (test_type, "waiting", DATASET_STATUS_NORMAL): 1,
        (test_other_type, "waiting", DATASET_STATUS_NORMAL): 1,
    }


def test_get_jobs_count_by_worker_size() -> None:
    test_type = "test_type"
    test_other_type = "test_other_type"
    test_dataset = "test_dataset"
    test_blocked_dataset = "blocked_dataset"
    test_revision = "test_revision"
    queue = Queue()

    assert queue.get_jobs_count_by_worker_size() == {"heavy": 0, "medium": 0, "light": 0}
    block_dataset(test_blocked_dataset)
    queue.add_job(job_type=test_type, dataset=test_blocked_dataset, revision=test_revision, difficulty=50)
    assert queue.get_jobs_count_by_worker_size() == {"heavy": 0, "medium": 0, "light": 0}
    queue.add_job(job_type=test_type, dataset=test_dataset, revision=test_revision, difficulty=50)
    assert queue.get_jobs_count_by_worker_size() == {"heavy": 0, "medium": 1, "light": 0}
    queue.add_job(job_type=test_other_type, dataset=test_dataset, revision=test_revision, difficulty=10)
    assert queue.get_jobs_count_by_worker_size() == {"heavy": 0, "medium": 1, "light": 1}
    block_dataset(test_dataset)
    assert queue.get_jobs_count_by_worker_size() == {"heavy": 0, "medium": 0, "light": 0}


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
                queue.finish_job(job_info["job_id"])
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
    queue.start_job()
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
    with patch("libcommon.queue.jobs.get_datetime", get_old_datetime):
        zombie = queue.add_job(
            job_type=job_type,
            dataset="dataset1",
            revision="revision",
            config="config",
            split="split1",
            difficulty=test_difficulty,
        )
        queue.start_job()
    queue.add_job(
        job_type=job_type,
        dataset="dataset1",
        revision="revision",
        config="config",
        split="split2",
        difficulty=test_difficulty,
    )
    queue.start_job()
    assert queue.get_zombies(max_seconds_without_heartbeat=10) == [zombie.reload().info()]
    assert queue.get_zombies(max_seconds_without_heartbeat=-1) == []
    assert queue.get_zombies(max_seconds_without_heartbeat=0) == []
    assert queue.get_zombies(max_seconds_without_heartbeat=9999999) == []


def test_delete_dataset_waiting_jobs() -> None:
    """
    Test that delete_dataset_waiting_jobs deletes all the waiting jobs for a dataset

    -> deletes at several levels (dataset, config, split)
    -> deletes waiting jobs, but not started jobs
    -> remove locks
    -> does not delete, and does not remove locks, for other datasets
    """
    dataset = "dataset"
    other_dataset = "other_dataset"
    job_type_1 = "job_type_1"
    job_type_2 = "job_type_2"
    job_type_3 = "job_type_3"
    job_type_4 = "job_type_4"
    revision = "dataset_git_revision"
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

    assert queue.delete_dataset_waiting_jobs(dataset=dataset) == 3

    assert JobDocument.objects().count() == 3
    assert JobDocument.objects(dataset=dataset).count() == 1
    assert JobDocument.objects(dataset=other_dataset).count() == 2
    assert JobDocument.objects(dataset=other_dataset, status=Status.STARTED).count() == 1
    assert JobDocument.objects(dataset=other_dataset, status=Status.WAITING).count() == 1


DATASET_1 = "dataset_1"
DATASET_2 = "dataset_2"
DATASET_3 = "dataset_3"
JOB_TYPE_1 = "job_type_1"
JOB_TYPE_2 = "job_type_2"
JOB_TYPE_3 = "job_type_3"
ALL_JOB_TYPES = {JOB_TYPE_1, JOB_TYPE_2, JOB_TYPE_3}
JOB_TYPE_4 = "job_type_4"


def create_jobs(queue: Queue) -> None:
    for dataset in [DATASET_1, DATASET_2]:
        for job_type in ALL_JOB_TYPES:
            queue.add_job(
                job_type=job_type,
                dataset=dataset,
                revision="dataset_git_revision",
                config=None,
                split=None,
                difficulty=50,
            )


@pytest.mark.parametrize(
    "dataset,job_types,expected_job_types",
    [
        (DATASET_1, None, ALL_JOB_TYPES),
        (DATASET_1, [JOB_TYPE_1], {JOB_TYPE_1}),
        (DATASET_1, [JOB_TYPE_1, JOB_TYPE_2], {JOB_TYPE_1, JOB_TYPE_2}),
        (DATASET_1, [JOB_TYPE_1, JOB_TYPE_4], {JOB_TYPE_1}),
        (DATASET_2, None, ALL_JOB_TYPES),
        (DATASET_3, None, set()),
    ],
)
def test_get_pending_jobs_df_and_has_pending_jobs(
    dataset: str,
    job_types: Optional[list[str]],
    expected_job_types: set[str],
) -> None:
    queue = Queue()
    create_jobs(queue)

    assert queue.has_pending_jobs(dataset=dataset, job_types=job_types) == bool(expected_job_types)

    df = queue.get_pending_jobs_df(dataset=dataset, job_types=job_types)
    assert len(df) == len(expected_job_types)
    if expected_job_types:
        assert df["dataset"].unique() == [dataset]
        assert set(df["type"].unique()) == set(expected_job_types)


@pytest.mark.parametrize(
    "datasets,blocked_datasets,expected_started_dataset",
    [
        ([], [], None),
        ([], ["dataset"], None),
        (["dataset"], [], "dataset"),
        (["dataset"], ["dataset"], None),
        (["dataset", "dataset"], [], "dataset"),
        (["dataset", "dataset"], ["dataset"], None),
        (["dataset1", "dataset2"], [], "dataset1"),
        (["dataset1", "dataset2"], ["dataset1"], "dataset2"),
        (["dataset1", "dataset2"], ["dataset1", "dataset2"], None),
    ],
)
def test_rate_limited_dataset(
    datasets: list[str],
    blocked_datasets: list[str],
    expected_started_dataset: Optional[str],
) -> None:
    queue = Queue()
    for dataset in datasets:
        queue.add_job(job_type="test_type", dataset=dataset, revision="test_revision", difficulty=50)
    for blocked_dataset in blocked_datasets:
        block_dataset(blocked_dataset)

    if expected_started_dataset:
        job_info = queue.start_job()
        assert job_info["params"]["dataset"] == expected_started_dataset
    else:
        with pytest.raises(EmptyQueueError):
            queue.start_job()
