# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.
# type: ignore
import posixpath
import shutil
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import fsspec
from fsspec.implementations.local import (
    AbstractFileSystem,
    LocalFileSystem,
    stringify_path,
)
from libapi.config import UvicornConfig
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import _clean_queue_database
from libcommon.resources import CacheMongoResource, QueueMongoResource
from libcommon.simple_cache import _clean_cache_database
from libcommon.storage import StrPath, init_cached_assets_dir, init_parquet_metadata_dir
from pytest import MonkeyPatch, fixture

from api.config import AppConfig, EndpointConfig, FilterAppConfig
from api.routes.endpoint import EndpointsDefinition, StepsByInputTypeAndEndpoint


# see https://github.com/pytest-dev/pytest/issues/363#issuecomment-406536200
@fixture(scope="session")
def monkeypatch_session() -> Iterator[MonkeyPatch]:
    monkeypatch_session = MonkeyPatch()
    monkeypatch_session.setenv("CACHE_MONGO_DATABASE", "datasets_server_cache_test")
    monkeypatch_session.setenv("QUEUE_MONGO_DATABASE", "datasets_server_queue_test")
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
        processing_step_names_by_input_type_and_endpoint={
            "/splits": {
                "dataset": ["dataset-split-names"],
                "config": ["config-split-names-from-streaming"],
            },
            "/first-rows": {"split": ["split-first-rows-from-streaming"]},
            "/parquet": {"config": ["config-parquet"]},
        }
    )


@fixture(scope="session")
def processing_graph(app_config: AppConfig) -> ProcessingGraph:
    return ProcessingGraph(app_config.processing_graph.specification)


@fixture(scope="session")
def endpoint_definition(
    endpoint_config: EndpointConfig, processing_graph: ProcessingGraph
) -> StepsByInputTypeAndEndpoint:
    return EndpointsDefinition(processing_graph, endpoint_config=endpoint_config).steps_by_input_type_and_endpoint


@fixture(scope="session")
def first_dataset_endpoint(endpoint_definition: StepsByInputTypeAndEndpoint) -> str:
    return next(
        endpoint
        for endpoint, input_types in endpoint_definition.items()
        if next((endpoint for input_type, _ in input_types.items() if input_type == "dataset"), None)
    )


@fixture(scope="session")
def first_config_endpoint(endpoint_definition: StepsByInputTypeAndEndpoint) -> str:
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


# TODO: duplicated in /rows
@fixture(scope="session")
def filter_app_config(monkeypatch_session: MonkeyPatch) -> FilterAppConfig:
    app_config = FilterAppConfig.from_env()
    if "test" not in app_config.cache.mongo_database or "test" not in app_config.queue.mongo_database:
        raise ValueError("Test must be launched on a test mongo database")
    return app_config


# TODO: duplicated in /rows
@fixture
def cached_assets_directory(filter_app_config: FilterAppConfig) -> StrPath:
    return init_cached_assets_dir(filter_app_config.cached_assets.storage_directory)


# TODO: duplicated in /rows
@fixture
def parquet_metadata_directory(filter_app_config: FilterAppConfig) -> StrPath:
    return init_parquet_metadata_dir(filter_app_config.parquet_metadata.storage_directory)


# TODO: duplicated in /rows
class MockFileSystem(AbstractFileSystem):
    protocol = "mock"

    def __init__(self, *args, local_root_dir, **kwargs):
        super().__init__()
        self._fs = LocalFileSystem(*args, **kwargs)
        self.local_root_dir = Path(local_root_dir).resolve().as_posix() + "/"

    def mkdir(self, path, *args, **kwargs):
        path = posixpath.join(self.local_root_dir, self._strip_protocol(path))
        return self._fs.mkdir(path, *args, **kwargs)

    def makedirs(self, path, *args, **kwargs):
        path = posixpath.join(self.local_root_dir, self._strip_protocol(path))
        return self._fs.makedirs(path, *args, **kwargs)

    def rmdir(self, path):
        path = posixpath.join(self.local_root_dir, self._strip_protocol(path))
        return self._fs.rmdir(path)

    def ls(self, path, detail=True, *args, **kwargs):
        path = posixpath.join(self.local_root_dir, self._strip_protocol(path))
        out = self._fs.ls(path, detail=detail, *args, **kwargs)
        if detail:
            return [{**info, "name": info["name"][len(self.local_root_dir) :]} for info in out]  # noqa: E203
        else:
            return [name[len(self.local_root_dir) :] for name in out]  # noqa: E203

    def info(self, path, *args, **kwargs):
        path = posixpath.join(self.local_root_dir, self._strip_protocol(path))
        out = dict(self._fs.info(path, *args, **kwargs))
        out["name"] = out["name"][len(self.local_root_dir) :]  # noqa: E203
        return out

    def cp_file(self, path1, path2, *args, **kwargs):
        path1 = posixpath.join(self.local_root_dir, self._strip_protocol(path1))
        path2 = posixpath.join(self.local_root_dir, self._strip_protocol(path2))
        return self._fs.cp_file(path1, path2, *args, **kwargs)

    def rm_file(self, path, *args, **kwargs):
        path = posixpath.join(self.local_root_dir, self._strip_protocol(path))
        return self._fs.rm_file(path, *args, **kwargs)

    def rm(self, path, *args, **kwargs):
        path = posixpath.join(self.local_root_dir, self._strip_protocol(path))
        return self._fs.rm(path, *args, **kwargs)

    def _open(self, path, *args, **kwargs):
        path = posixpath.join(self.local_root_dir, self._strip_protocol(path))
        return self._fs._open(path, *args, **kwargs)

    def created(self, path):
        path = posixpath.join(self.local_root_dir, self._strip_protocol(path))
        return self._fs.created(path)

    def modified(self, path):
        path = posixpath.join(self.local_root_dir, self._strip_protocol(path))
        return self._fs.modified(path)

    @classmethod
    def _strip_protocol(cls, path):
        path = stringify_path(path)
        if path.startswith("mock://"):
            path = path[7:]
        return path


# TODO: duplicated in /rows
class TmpDirFileSystem(MockFileSystem):
    protocol = "tmp"
    tmp_dir = None

    def __init__(self, *args, **kwargs):
        assert self.tmp_dir is not None, "TmpDirFileSystem.tmp_dir is not set"
        super().__init__(*args, **kwargs, local_root_dir=self.tmp_dir, auto_mkdir=True)

    @classmethod
    def _strip_protocol(cls, path):
        path = stringify_path(path)
        if path.startswith("tmp://"):
            path = path[6:]
        return path


# TODO: duplicated in /rows
@fixture
def mock_fsspec():
    original_registry = fsspec.registry.copy()
    fsspec.register_implementation("mock", MockFileSystem)
    fsspec.register_implementation("tmp", TmpDirFileSystem)
    yield
    fsspec.registry = original_registry


# # TODO: duplicated in /rows
# @fixture
# def mockfs(tmp_path_factory, mock_fsspec):
#     local_fs_dir = tmp_path_factory.mktemp("mockfs")
#     return MockFileSystem(local_root_dir=local_fs_dir, auto_mkdir=True)


# TODO: duplicated in /rows
@fixture
def tmpfs(tmp_path_factory, mock_fsspec):
    tmp_fs_dir = tmp_path_factory.mktemp("tmpfs")
    with patch.object(TmpDirFileSystem, "tmp_dir", tmp_fs_dir):
        yield TmpDirFileSystem()
    shutil.rmtree(tmp_fs_dir)
