# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Iterator

from pytest import MonkeyPatch, fixture, mark
from starlette.testclient import TestClient

from api.app import create_app
from api.config import AppConfig

API_HF_WEBHOOK_SECRET = "some secret"


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="module")
def real_monkeypatch() -> Iterator[MonkeyPatch]:
    monkeypatch = MonkeyPatch()
    monkeypatch.setenv("CACHE_MONGO_DATABASE", "dataset_viewer_cache_test")
    monkeypatch.setenv("QUEUE_MONGO_DATABASE", "dataset_viewer_queue_test")
    monkeypatch.setenv("COMMON_HF_ENDPOINT", "https://huggingface.co")
    monkeypatch.setenv("COMMON_HF_TOKEN", "")
    monkeypatch.setenv("API_HF_WEBHOOK_SECRET", API_HF_WEBHOOK_SECRET)
    yield monkeypatch
    monkeypatch.undo()


@fixture(scope="module")
def real_client(real_monkeypatch: MonkeyPatch) -> TestClient:
    return TestClient(create_app())


@fixture(scope="module")
def real_app_config(real_monkeypatch: MonkeyPatch) -> AppConfig:
    app_config = AppConfig.from_env()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    if app_config.common.hf_endpoint != "https://huggingface.co":
        raise ValueError("Test must be launched on the production hub")
    return app_config


@mark.real_dataset
def test_webhook_untrusted(
    real_client: TestClient,
) -> None:
    payload = {
        "event": "add",
        "repo": {"type": "dataset", "name": "nyu-mll/glue", "gitalyUid": "123", "headSha": "revision"},
        "scope": "repo",
    }
    response = real_client.post("/webhook", json=payload)
    assert response.status_code == 400, response.text


@mark.real_dataset
def test_webhook_trusted(real_client: TestClient) -> None:
    payload = {
        "event": "add",
        "repo": {"type": "dataset", "name": "nyu-mll/glue", "gitalyUid": "123", "headSha": "revision"},
        "scope": "repo",
    }
    response = real_client.post("/webhook", json=payload, headers={"x-webhook-secret": API_HF_WEBHOOK_SECRET})
    assert response.status_code == 200, response.text
