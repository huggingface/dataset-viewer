# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

from copy import deepcopy
from http import HTTPStatus
from unittest.mock import AsyncMock, Mock

import pytest
from libapi.exceptions import ExternalUnauthenticatedError
from libcommon.processing_graph import processing_graph
from libcommon.storage_client import StorageClient
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

from api.config import EndpointConfig
from api.routes.endpoint import EndpointsDefinition, create_endpoint


@pytest.fixture
def cache_mongo_resource() -> None:
    pass


@pytest.fixture
def queue_mongo_resource() -> None:
    pass


@pytest.fixture
def cached_response() -> Mock:
    return Mock(
        return_value={
            "content": {
                "dataset": "org/harbor",
                "revision": "commit-sha",
                "tasks": [
                    {"path": path, "files": [{"path": f"{path + '/' if path else ''}instruction.md", "size": 42}]}
                    for path in ["", *(f"tasks/{index:03}" for index in range(1, 205))]
                ],
            },
            "http_status": HTTPStatus.OK,
            "error_code": None,
            "dataset_git_revision": "commit-sha",
        }
    )


@pytest.fixture
def auth_check() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, cached_response: Mock, auth_check: AsyncMock) -> TestClient:
    monkeypatch.setattr("api.routes.endpoint.auth_check", auth_check)
    monkeypatch.setattr("api.routes.endpoint.get_cache_entry_from_step", cached_response)
    definition = EndpointsDefinition(processing_graph, EndpointConfig())
    endpoint = create_endpoint(
        endpoint_name="/harbor-tasks",
        step_by_input_type=definition.step_by_input_type_and_endpoint["/harbor-tasks"],
        hf_endpoint="https://huggingface.co",
        blocked_datasets=[],
        assets_storage_client=Mock(spec=StorageClient),
        max_age_long=120,
        max_age_short=10,
    )
    return TestClient(Starlette(routes=[Route("/harbor-tasks", endpoint=endpoint)]))


def test_harbor_tasks_pagination(client: TestClient, cached_response: Mock) -> None:
    original = deepcopy(cached_response.return_value)
    response = client.get("/harbor-tasks", params={"dataset": "org/harbor"})
    assert response.status_code == 200
    assert response.json() == {
        **original["content"],
        "tasks": original["content"]["tasks"][:100],
        "num_tasks_total": 205,
        "offset": 0,
        "length": 100,
    }
    assert response.headers["X-Revision"] == "commit-sha"
    assert response.headers["Cache-Control"] == "max-age=120"
    response = client.get("/harbor-tasks", params={"dataset": "org/harbor", "offset": 200, "length": 10})
    assert response.json()["tasks"] == original["content"]["tasks"][200:]
    assert response.json()["offset"] == 200
    assert response.json()["length"] == 10
    assert cached_response.return_value == original
    assert cached_response.call_args.kwargs["processing_step_name"] == "dataset-harbor-tasks"
    assert cached_response.call_args.kwargs["config"] is None
    assert cached_response.call_args.kwargs["split"] is None


@pytest.mark.parametrize("task,offset", [("tasks/204", 200), ("", 0)])
def test_harbor_tasks_deep_link(client: TestClient, task: str, offset: int) -> None:
    response = client.get("/harbor-tasks", params={"dataset": "org/harbor", "offset": 100, "task": task})
    assert response.status_code == 200
    assert response.json()["offset"] == offset
    assert any(item["path"] == task for item in response.json()["tasks"])


def test_harbor_task_not_found(client: TestClient) -> None:
    response = client.get("/harbor-tasks", params={"dataset": "org/harbor", "task": "missing"})
    assert response.status_code == 404
    assert response.headers["X-Error-Code"] == "ResponseNotFound"
    assert response.headers["X-Revision"] == "commit-sha"


@pytest.mark.parametrize(
    "params", [{}, {"offset": "-1"}, {"offset": "x"}, {"length": "0"}, {"length": "101"}, {"length": "x"}]
)
def test_harbor_tasks_invalid_parameters(client: TestClient, cached_response: Mock, params: dict[str, str]) -> None:
    response = client.get("/harbor-tasks", params={"dataset": "org/harbor", **params} if params else {})
    assert response.status_code == 422
    cached_response.assert_not_called()


def test_harbor_tasks_auth_before_cache(client: TestClient, cached_response: Mock, auth_check: AsyncMock) -> None:
    auth_check.side_effect = ExternalUnauthenticatedError("Authentication required")
    response = client.get("/harbor-tasks", params={"dataset": "org/harbor"})
    assert response.status_code == 401
    assert response.headers["X-Error-Code"] == "ExternalUnauthenticatedError"
    cached_response.assert_not_called()


def test_harbor_tasks_cached_error(client: TestClient, cached_response: Mock) -> None:
    cached_response.return_value.update(
        content={"error": "Not a Harbor dataset"}, http_status=HTTPStatus.NOT_FOUND, error_code="DatasetNotSupported"
    )
    response = client.get("/harbor-tasks", params={"dataset": "org/harbor"})
    assert response.status_code == 404
    assert response.json() == {"error": "Not a Harbor dataset"}
    assert response.headers["X-Error-Code"] == "DatasetNotSupported"
    assert response.headers["Cache-Control"] == "max-age=10"
    assert response.headers["X-Revision"] == "commit-sha"


def test_harbor_tasks_empty_page(client: TestClient, cached_response: Mock) -> None:
    cached_response.return_value["content"]["tasks"] = []
    response = client.get("/harbor-tasks", params={"dataset": "org/harbor"})
    assert response.status_code == 200
    assert response.json()["tasks"] == []
    assert response.json()["num_tasks_total"] == 0
