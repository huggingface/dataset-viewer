# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import uvicorn
from libapi.config import UvicornConfig
from libapi.jwt_token import get_jwt_public_keys
from libapi.routes.healthcheck import healthcheck_endpoint
from libapi.routes.metrics import create_metrics_endpoint
from libapi.utils import EXPOSED_HEADERS
from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource, Resource
from libcommon.s3_client import S3Client
from libcommon.storage import exists, init_cached_assets_dir, init_parquet_metadata_dir
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from rows.config import AppConfig
from rows.routes.rows import create_rows_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()
    return create_app_with_config(app_config=app_config)


def create_app_with_config(app_config: AppConfig) -> Starlette:
    init_logging(level=app_config.log.level)
    # ^ set first to have logs as soon as possible
    cached_assets_directory = init_cached_assets_dir(directory=app_config.cached_assets.storage_directory)
    if not exists(cached_assets_directory):
        raise RuntimeError("The assets storage directory could not be accessed. Exiting.")
    parquet_metadata_directory = init_parquet_metadata_dir(directory=app_config.parquet_metadata.storage_directory)
    if not exists(parquet_metadata_directory):
        raise RuntimeError("The parquet metadata storage directory could not be accessed. Exiting.")

    processing_graph = ProcessingGraph(app_config.processing_graph)
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
    s3_client = S3Client(
        aws_access_key_id=app_config.s3.access_key_id,
        aws_secret_access_key=app_config.s3.secret_access_key,
        region_name=app_config.s3.region,
        bucket_name=app_config.s3.bucket,
    )
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
            "/rows",
            endpoint=create_rows_endpoint(
                processing_graph=processing_graph,
                cached_assets_base_url=app_config.cached_assets.base_url,
                cached_assets_directory=cached_assets_directory,
                cached_assets_s3_folder_name=app_config.cached_assets.s3_folder_name,
                s3_client=s3_client,
                parquet_metadata_directory=parquet_metadata_directory,
                max_arrow_data_in_memory=app_config.rows_index.max_arrow_data_in_memory,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                blocked_datasets=app_config.common.blocked_datasets,
                hf_jwt_public_keys=hf_jwt_public_keys,
                hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
                external_auth_url=app_config.api.external_auth_url,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
                cache_max_days=app_config.cache.max_days,
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
