# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import pytest

from .utils import poll

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.files", "tests.fixtures.hub"]


@pytest.fixture(autouse=True, scope="session")
def ensure_services_are_up() -> None:
    assert poll("/", expected_code=404).status_code == 404


# MOVED FROM API - to implement as e2e tests


# # caveat: the returned status codes don't simulate the reality
# # they're just used to check every case
# @pytest.mark.parametrize(
#     "headers,status_code,error_code",
#     [
#         ({"Cookie": "some cookie"}, 401, "ExternalUnauthenticatedError"),
#         ({"Authorization": "Bearer invalid"}, 404, "ExternalAuthenticatedError"),
#         ({}, 500, "ResponseNotReady"),
#     ],
# )
# def test_splits_auth(
#     client: TestClient,
#     httpserver: HTTPServer,
#     hf_auth_path: str,
#     hf_ask_access_path: str,
#     headers: Mapping[str, str],
#     status_code: int,
#     error_code: str,
#     first_dataset_processing_step: ProcessingStep,
# ) -> None:
#     dataset = "dataset-which-does-not-exist"
#     httpserver.expect_request(hf_auth_path % dataset, headers=headers).respond_with_handler(auth_callback)
#     httpserver.expect_request(hf_ask_access_path % dataset, headers=headers).respond_with_data("ok", status=200)
#     # ^TODO
#     httpserver.expect_request(f"/api/datasets/{dataset}").respond_with_data(
#         json.dumps({}), headers={"X-Error-Code": "RepoNotFound"}
#     )
#     response = client.get(f"{first_dataset_processing_step.endpoint}?dataset={dataset}", headers=headers)
#     assert response.status_code == status_code, f"{response.headers}, {response.json()}"
#     assert response.headers.get("X-Error-Code") == error_code


# @pytest.mark.parametrize(
#     "exists,is_private,expected_error_code",
#     [
#         (False, None, "ExternalAuthenticatedError"),
#         (True, True, "ResponseNotFound"),
#         (True, False, "ResponseNotReady"),
#     ],
# )
# def test_cache_refreshing(
#     client: TestClient,
#     httpserver: HTTPServer,
#     hf_auth_path: str,
#     exists: bool,
#     is_private: Optional[bool],
#     expected_error_code: str,
#     app_config: AppConfig,
# ) -> None:
#     dataset = "dataset-to-be-processed"
#     for step in app_config.processing_graph.graph.steps.values():
#         config = None if step.input_type == "dataset" else "config"
#         split = None if step.input_type == "dataset" else "split"
#         httpserver.expect_request(hf_auth_path % dataset).respond_with_data(status=200 if exists else 404)
#         httpserver.expect_request(f"/api/datasets/{dataset}").respond_with_data(
#             json.dumps({"private": is_private}), headers={} if exists else {"X-Error-Code": "RepoNotFound"}
#         )

#         response = client.get(step.endpoint, params={"dataset": dataset, "config": config, "split": split})
#         assert response.headers["X-Error-Code"] == expected_error_code

#         if expected_error_code == "ResponseNotReady":
#             # a subsequent request should return the same error code
#             response = client.get(step.endpoint, params={"dataset": dataset, "config": config, "split": split})
#             assert response.headers["X-Error-Code"] == expected_error_code

#             # simulate the worker
#             upsert_response(
#                 kind=step.cache_kind,
#                 dataset=dataset,
#                 config=config,
#                 split=split,
#                 content={"key": "value"},
#                 http_status=HTTPStatus.OK,
#             )
#             response = client.get(step.endpoint, params={"dataset": dataset, "config": config, "split": split})
#             assert response.json()["key"] == "value"
#             assert response.status_code == 200


# @pytest.mark.parametrize(
#     "payload,exists_on_the_hub,expected_status,expected_is_updated",
#     [
#         ({"event": "add", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}}, True, 200, True),
#         (
#             {
#                 "event": "move",
#                 "movedTo": "webhook-test",
#                 "repo": {"type": "dataset", "name": "previous-name", "gitalyUid": "123"},
#             },
#             True,
#             200,
#             True,
#         ),
#         (
#             {"event": "doesnotexist", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}},
#             True,
#             400,
#             False,
#         ),
#         (
#             {"event": "add", "repo": {"type": "dataset", "name": "webhook-test"}},
#             True,
#             200,
#             True,
#         ),
#         (
#             {"event": "add", "repo": {"type": "dataset", "name": "webhook-test", "gitalyUid": "123"}},
#             False,
#             400,
#             False
#         ),
#     ],
# )
# def test_webhook(
#     client: TestClient,
#     httpserver: HTTPServer,
#     payload: Mapping[str, Any],
#     exists_on_the_hub: bool,
#     expected_status: int,
#     expected_is_updated: bool,
#     app_config: AppConfig,
# ) -> None:
#     dataset = "webhook-test"
#     headers = None if exists_on_the_hub else {"X-Error-Code": "RepoNotFound"}
#     status = 200 if exists_on_the_hub else 404
#     httpserver.expect_request(f"/api/datasets/{dataset}").respond_with_data(
#         json.dumps({"private": False}), headers=headers, status=status
#     )
#     response = client.post("/webhook", json=payload)
#     assert response.status_code == expected_status, response.text
#     for step in app_config.processing_graph.graph.steps.values():
#         if not step.parent:
#             assert Queue(type=step.job_type).is_job_in_process(dataset=dataset) is expected_is_updated
#         else:
#             assert Queue(type=step.job_type).is_job_in_process(dataset=dataset) is False
