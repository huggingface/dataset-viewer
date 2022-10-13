# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from libqueue.queue import (
    EmptyQueue,
    Job,
    Status,
    _clean_queue_database,
    add_job,
    connect_to_queue,
    finish_job,
    get_datetime,
    get_jobs_count_by_status,
    is_job_in_process,
    start_job,
)

from ._utils import MONGO_QUEUE_DATABASE, MONGO_URL


@pytest.fixture(autouse=True, scope="module")
def safe_guard() -> None:
    if "test" not in MONGO_QUEUE_DATABASE:
        raise ValueError("Test must be launched on a test mongo database")


@pytest.fixture(autouse=True, scope="module")
def client() -> None:
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)


@pytest.fixture(autouse=True)
def clean_mongo_database() -> None:
    _clean_queue_database()


def test_add_job() -> None:
    test_type = "test_type"
    test_dataset = "test_dataset"
    # add a job
    add_job(type=test_type, dataset=test_dataset)
    # a second call is ignored
    add_job(type=test_type, dataset=test_dataset)
    assert is_job_in_process(type=test_type, dataset=test_dataset) is True
    # get and start the first job
    job_id, dataset, config, split = start_job(type=test_type)
    assert dataset == test_dataset
    assert config is None
    assert split is None
    assert is_job_in_process(type=test_type, dataset=test_dataset) is True
    # adding the job while the first one has not finished yet is ignored
    add_job(type=test_type, dataset=test_dataset)
    with pytest.raises(EmptyQueue):
        # thus: no new job available
        start_job(type=test_type)
    # finish the first job
    finish_job(job_id=job_id, success=True)
    # the queue is empty
    assert is_job_in_process(type=test_type, dataset=test_dataset) is False
    with pytest.raises(EmptyQueue):
        start_job(type=test_type)
    # add a job again
    add_job(type=test_type, dataset=test_dataset)
    # start it
    job_id, *_ = start_job(type=test_type)
    other_job_id = ("1" if job_id[0] == "0" else "0") + job_id[1:]
    finish_job(job_id=other_job_id, success=True)
    # ^ fails silently (with a log)
    finish_job(job_id=job_id, success=True)


def test_add_job_with_broken_collection() -> None:
    test_type = "test_type"
    test_dataset = "dataset_broken"
    test_config = "config_broken"
    test_split = "split_broken"
    # ensure the jobs are cancelled with more than one exist in a "pending" status
    # we "manually" create two jobs in a "pending" status for the same split
    # (we normally cannot do that with the exposed methods)
    job_1 = Job(
        type=test_type,
        dataset=test_dataset,
        config=test_config,
        split=test_split,
        created_at=get_datetime(),
        status=Status.WAITING,
    ).save()
    job_2 = Job(
        type=test_type,
        dataset=test_dataset,
        config=test_config,
        split=test_split,
        created_at=get_datetime(),
        started_at=get_datetime(),
        status=Status.STARTED,
    ).save()
    # then we add a job: it should create a new job in the "WAITING" status
    # and the two other jobs should be cancelled
    add_job(type=test_type, dataset=test_dataset, config=test_config, split=test_split)
    assert (
        Job.objects(
            type=test_type, dataset=test_dataset, config=test_config, split=test_split, status__in=[Status.WAITING]
        ).count()
        == 1
    )
    assert Job.objects(pk=job_1.pk).get().status == Status.CANCELLED
    assert Job.objects(pk=job_2.pk).get().status == Status.CANCELLED


def test_priority_to_non_started_datasets() -> None:
    test_type = "test_type"
    add_job(type=test_type, dataset="dataset1", config="config", split="split1")
    add_job(type=test_type, dataset="dataset1", config="config", split="split2")
    add_job(type=test_type, dataset="dataset1", config="config", split="split3")
    add_job(type=test_type, dataset="dataset2", config="config", split="split1")
    add_job(type=test_type, dataset="dataset2", config="config", split="split2")
    add_job(type=test_type, dataset="dataset3", config="config", split="split1")
    _, dataset, __, split = start_job(type=test_type)
    assert dataset == "dataset1"
    assert split == "split1"
    _, dataset, __, split = start_job(type=test_type)
    assert dataset == "dataset2"
    assert split == "split1"
    _, dataset, __, split = start_job(type=test_type)
    assert dataset == "dataset3"
    assert split == "split1"
    _, dataset, __, split = start_job(type=test_type)
    assert dataset == "dataset1"
    assert split == "split2"
    _, dataset, __, split = start_job(type=test_type)
    assert dataset == "dataset1"
    assert split == "split3"
    _, dataset, __, split = start_job(type=test_type)
    assert dataset == "dataset2"
    assert split == "split2"
    with pytest.raises(EmptyQueue):
        start_job(type=test_type)


def test_max_jobs_per_dataset() -> None:
    test_type = "test_type"
    test_dataset = "test_dataset"
    test_config = "test_config"
    add_job(type=test_type, dataset=test_dataset, config=test_config, split="split1")
    assert is_job_in_process(type=test_type, dataset=test_dataset, config=test_config, split="split1") is True
    add_job(type=test_type, dataset=test_dataset, config=test_config, split="split2")
    add_job(type=test_type, dataset=test_dataset, config=test_config, split="split3")
    job_id, dataset, config, split = start_job(type=test_type)
    assert dataset == test_dataset
    assert config == test_config
    assert split == "split1"
    assert is_job_in_process(type=test_type, dataset=test_dataset, config=test_config, split="split1") is True
    with pytest.raises(EmptyQueue):
        start_job(type=test_type, max_jobs_per_dataset=0)
    with pytest.raises(EmptyQueue):
        start_job(type=test_type, max_jobs_per_dataset=1)
    _, dataset, config, split = start_job(type=test_type, max_jobs_per_dataset=2)
    assert split == "split2"
    with pytest.raises(EmptyQueue):
        start_job(type=test_type, max_jobs_per_dataset=2)
    # finish the first job
    finish_job(job_id, success=True)
    assert is_job_in_process(type=test_type, dataset=test_dataset, config=test_config, split="split1") is False


def test_count_by_status() -> None:
    test_type = "test_type"
    test_other_type = "test_other_type"
    test_dataset = "test_dataset"

    expected_empty = {"waiting": 0, "started": 0, "success": 0, "error": 0, "cancelled": 0}
    expected_one_waiting = {"waiting": 1, "started": 0, "success": 0, "error": 0, "cancelled": 0}
    expected_two_waiting = {"waiting": 2, "started": 0, "success": 0, "error": 0, "cancelled": 0}

    assert get_jobs_count_by_status() == expected_empty

    add_job(type=test_type, dataset=test_dataset)

    assert get_jobs_count_by_status() == expected_one_waiting
    assert get_jobs_count_by_status(type=test_type) == expected_one_waiting
    assert get_jobs_count_by_status(type=test_other_type) == expected_empty

    add_job(type=test_other_type, dataset=test_dataset)

    assert get_jobs_count_by_status() == expected_two_waiting
    assert get_jobs_count_by_status(type=test_type) == expected_one_waiting
    assert get_jobs_count_by_status(type=test_other_type) == expected_one_waiting
