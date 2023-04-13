# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from pathlib import Path
from typing import Iterator

from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import _clean_queue_database
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import _clean_cache_database
from libcommon.storage import StrPath, init_cached_assets_dir
from pytest import MonkeyPatch, fixture

from api.config import AppConfig, EndpointConfig, UvicornConfig
from api.routes.endpoint import EndpointsDefinition, StepsByInputTypeAndEndpoint

# Import fixture modules as plugins
pytest_plugins = ["tests.fixtures.fsspec"]


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session() -> Iterator[MonkeyPatch]:
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
    monkeypatch_session.setenv("CACHED_ASSETS_BASE_URL", "http://localhost/cached-assets")
    hostname = "localhost"
    port = "8888"
    monkeypatch_session.setenv("API_HF_TIMEOUT_SECONDS", "10")
    monkeypatch_session.setenv("API_UVICORN_HOSTNAME", hostname)
    monkeypatch_session.setenv("API_UVICORN_PORT", port)
    monkeypatch_session.setenv("COMMON_HF_ENDPOINT", f"http://{hostname}:{port}")
    yield monkeypatch_session
    monkeypatch_session.undo()


@fixture(scope="session")
def app_config(monkeypatch_session: MonkeyPatch) -> AppConfig:
    app_config = AppConfig.from_env()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return app_config


@fixture(scope="session")
def endpoint_config(monkeypatch_session: MonkeyPatch) -> EndpointConfig:
    return EndpointConfig(
        step_names_by_input_type_and_endpoint={
            "/config-names": {"dataset": ["/config-names"]},
            "/splits": {
                "config": ["/split-names-from-streaming"],
            },
            "/first-rows": {"split": ["split-first-rows-from-streaming"]},
            "/parquet": {"config": ["config-parquet"]},
        }
    )


@fixture(scope="session")
def endpoint_definition(endpoint_config: EndpointConfig, app_config: AppConfig) -> StepsByInputTypeAndEndpoint:
    processing_graph = ProcessingGraph(app_config.processing_graph.specification)
    return EndpointsDefinition(processing_graph, endpoint_config=endpoint_config).steps_by_input_type_and_endpoint


@fixture(scope="session")
def first_dataset_endpoint(endpoint_definition: StepsByInputTypeAndEndpoint) -> str:
    return next(
        endpoint
        for endpoint, input_types in endpoint_definition.items()
        if next((endpoint for input_type, _ in input_types.items() if input_type == "dataset"), None)
    )


@fixture(scope="session")
def first_config_endoint(endpoint_definition: StepsByInputTypeAndEndpoint) -> str:
    return next(
        endpoint
        for endpoint, input_types in endpoint_definition.items()
        if next((endpoint for input_type, _ in input_types.items() if input_type == "config"), None)
    )


@fixture(scope="session")
def first_split_endpoint(endpoint_definition: StepsByInputTypeAndEndpoint) -> str:
    return next(
        endpoint
        for endpoint, input_types in endpoint_definition.items()
        if next((endpoint for input_type, _ in input_types.items() if input_type == "split"), None)
    )


@fixture(autouse=True)
def cache_mongo_resource(app_config: AppConfig) -> Iterator[CacheMongoResource]:
    with CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url) as resource:
        yield resource
        _clean_cache_database()


@fixture(autouse=True)
def queue_mongo_resource(app_config: AppConfig) -> Iterator[QueueMongoResource]:
    with QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url) as resource:
        yield resource
        _clean_queue_database()


@fixture(scope="session")
def uvicorn_config(monkeypatch_session: MonkeyPatch) -> UvicornConfig:
    return UvicornConfig.from_env()


@fixture(scope="session")
def httpserver_listen_address(uvicorn_config: UvicornConfig) -> tuple[str, int]:
    return (uvicorn_config.hostname, uvicorn_config.port)


@fixture(scope="session")
def hf_endpoint(app_config: AppConfig) -> str:
    return app_config.common.hf_endpoint


@fixture(scope="session")
def hf_auth_path(app_config: AppConfig) -> str:
    return app_config.api.hf_auth_path


@fixture
def cached_assets_directory(app_config: AppConfig) -> StrPath:
    return init_cached_assets_dir(app_config.cached_assets.storage_directory)


@fixture
def image_path() -> str:
    image_path = Path(__file__).resolve().parent / "data" / "test_image_rgb.jpg"
    assert image_path.is_file()
    return str(image_path)
