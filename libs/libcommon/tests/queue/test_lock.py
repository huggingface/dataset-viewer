# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import json
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path

import pytest

from libcommon.new_queue.lock import Lock, lock
from libcommon.resources import QueueMongoResource


@pytest.fixture(autouse=True)
def queue_mongo_resource_autouse(queue_mongo_resource: QueueMongoResource) -> QueueMongoResource:
    return queue_mongo_resource


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
