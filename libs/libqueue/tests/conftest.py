# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pytest import MonkeyPatch, fixture

from libqueue.config import QueueConfig


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session():
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(scope="session")
def queue_config(monkeypatch_session: MonkeyPatch) -> QueueConfig:
    queue_config = QueueConfig()
    if "test" not in queue_config.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return queue_config
