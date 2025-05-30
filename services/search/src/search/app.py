# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import uvicorn
from libapi.config import UvicornConfig
from libapi.jwt_token import get_jwt_public_keys
from libapi.routes.healthcheck import healthcheck_endpoint
from libapi.routes.metrics import create_metrics_endpoint
from libapi.utils import EXPOSED_HEADERS
from libcommon.cloudfront import get_cloudfront_signer
from libcommon.log import init_logging
from libcommon.resources import CacheMongoResource, QueueMongoResource, Resource
from libcommon.storage import exists, init_duckdb_index_cache_dir, init_parquet_metadata_dir
from libcommon.storage_client import StorageClient
from libcommon.url_preparator import URLPreparator
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from search.config import AppConfig
from search.routes.filter import create_filter_endpoint
from search.routes.search import create_search_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()
    return create_app_with_config(app_config=app_config)


def create_app_with_config(app_config: AppConfig) -> Starlette:
    init_logging(level=app_config.log.level)
    # ^ set first to have logs as soon as possible

    duckdb_index_cache_directory = init_duckdb_index_cache_dir(directory=app_config.duckdb_index.cache_directory)
    if not exists(duckdb_index_cache_directory):
        raise RuntimeError("The duckdb_index cache directory could not be accessed. Exiting.")
    parquet_metadata_directory = init_parquet_metadata_dir(directory=app_config.parquet_metadata.storage_directory)
    if not exists(parquet_metadata_directory):
        raise RuntimeError("The parquet metadata storage directory could not be accessed. Exiting.")

    hf_jwt_public_keys = get_jwt_public_keys(
        algorithm_name=app_config.api.hf_jwt_algorithm,
        public_key_url=app_config.api.hf_jwt_public_key_url,
        additional_public_keys=app_config.api.hf_jwt_additional_public_keys,
        timeout_seconds=app_config.api.hf_timeout_seconds,
    )

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
            expose_headers=EXPOSED_HEADERS,
        ),
        Middleware(GZipMiddleware),
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]

    cache_resource = CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url)
    queue_resource = QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url)
    url_signer = get_cloudfront_signer(cloudfront_config=app_config.cloudfront)
    url_preparator = URLPreparator(
        url_signer=url_signer,
        hf_endpoint=app_config.common.hf_endpoint,
        assets_base_url=app_config.cached_assets.base_url,
    )
    cached_assets_storage_client = StorageClient(
        protocol=app_config.cached_assets.storage_protocol,
        storage_root=app_config.cached_assets.storage_root,
        base_url=app_config.cached_assets.base_url,
        s3_config=app_config.s3,
        url_preparator=url_preparator,
    )
    assets_storage_client = StorageClient(
        protocol=app_config.assets.storage_protocol,
        storage_root=app_config.assets.storage_root,
        base_url=app_config.assets.base_url,
        s3_config=app_config.s3,
        # no need to specify a url_signer
    )
    storage_clients = [cached_assets_storage_client, assets_storage_client]
    resources: list[Resource] = [cache_resource, queue_resource]
    if not cache_resource.is_available():
        raise RuntimeError("The connection to the cache database could not be established. Exiting.")
    if not queue_resource.is_available():
        raise RuntimeError("The connection to the queue database could not be established. Exiting.")

    routes = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/metrics", endpoint=create_metrics_endpoint()),
        # ^ called by Prometheus
        Route(
            "/search",
            endpoint=create_search_endpoint(
                duckdb_index_file_directory=duckdb_index_cache_directory,
                cached_assets_storage_client=cached_assets_storage_client,
                parquet_metadata_directory=parquet_metadata_directory,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                blocked_datasets=app_config.common.blocked_datasets,
                hf_jwt_public_keys=hf_jwt_public_keys,
                hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
                external_auth_url=app_config.api.external_auth_url,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
                storage_clients=storage_clients,
                extensions_directory=app_config.duckdb_index.extensions_directory,
                clean_cache_proba=app_config.duckdb_index.clean_cache_proba,
                expiredTimeIntervalSeconds=app_config.duckdb_index.expired_time_interval_seconds,
            ),
        ),
        Route(
            "/filter",
            endpoint=create_filter_endpoint(
                duckdb_index_file_directory=duckdb_index_cache_directory,
                cached_assets_storage_client=cached_assets_storage_client,
                parquet_metadata_directory=parquet_metadata_directory,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                blocked_datasets=app_config.common.blocked_datasets,
                hf_jwt_public_keys=hf_jwt_public_keys,
                hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
                external_auth_url=app_config.api.external_auth_url,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
                storage_clients=storage_clients,
                extensions_directory=app_config.duckdb_index.extensions_directory,
                clean_cache_proba=app_config.duckdb_index.clean_cache_proba,
                expiredTimeIntervalSeconds=app_config.duckdb_index.expired_time_interval_seconds,
            ),
        ),
    ]

    return Starlette(routes=routes, middleware=middleware, on_shutdown=[resource.release for resource in resources])


def start() -> None:
    uvicorn_config = UvicornConfig.from_env()
    uvicorn.run(
        "app:create_app",
        host=uvicorn_config.hostname,
        port=uvicorn_config.port,
        factory=True,
        workers=uvicorn_config.num_workers,
    )
