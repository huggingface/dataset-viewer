# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import uvicorn
from libapi.utils import EXPOSED_HEADERS
from libcommon.log import init_logging
from libcommon.processing_graph import processing_graph
from libcommon.resources import CacheMongoResource, QueueMongoResource, Resource
from libcommon.storage import (
    init_parquet_metadata_dir,
)
from libcommon.storage_client import StorageClient
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Mount, Route
from starlette_prometheus import PrometheusMiddleware

from admin.config import AppConfig, UvicornConfig
from admin.routes.cache_reports import create_cache_reports_endpoint
from admin.routes.cache_reports_with_content import (
    create_cache_reports_with_content_endpoint,
)
from admin.routes.dataset_status import create_dataset_status_endpoint
from admin.routes.force_refresh import create_force_refresh_endpoint
from admin.routes.healthcheck import healthcheck_endpoint
from admin.routes.metrics import create_metrics_endpoint
from admin.routes.num_dataset_infos_by_builder_name import (
    create_num_dataset_infos_by_builder_name_endpoint,
)
from admin.routes.pending_jobs import create_pending_jobs_endpoint
from admin.routes.recreate_dataset import create_recreate_dataset_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()

    init_logging(level=app_config.log.level)
    # ^ set first to have logs as soon as possible
    parquet_metadata_directory = init_parquet_metadata_dir(directory=app_config.parquet_metadata.storage_directory)

    cached_assets_storage_client = StorageClient(
        protocol=app_config.cached_assets.storage_protocol,
        storage_root=app_config.cached_assets.storage_root,
        base_url=app_config.cached_assets.base_url,
        s3_config=app_config.s3,
        # no need to specify a url_signer
    )
    assets_storage_client = StorageClient(
        protocol=app_config.assets.storage_protocol,
        storage_root=app_config.assets.storage_root,
        base_url=app_config.assets.base_url,
        s3_config=app_config.s3,
        # no need to specify a url_signer
    )
    storage_clients = [cached_assets_storage_client, assets_storage_client]

    cache_resource = CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url)
    queue_resource = QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url)
    resources: list[Resource] = [cache_resource, queue_resource]
    if not cache_resource.is_available():
        raise RuntimeError("The connection to the cache database could not be established. Exiting.")
    if not queue_resource.is_available():
        raise RuntimeError("The connection to the queue database could not be established. Exiting.")

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
    routes = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route(
            "/metrics",
            endpoint=create_metrics_endpoint(
                parquet_metadata_directory=parquet_metadata_directory,
            ),
        ),
        # used in a browser tab to monitor the queue
        Route(
            "/pending-jobs",
            endpoint=create_pending_jobs_endpoint(
                max_age=app_config.admin.max_age,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
                hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
            ),
        ),
        Route(
            "/dataset-status",
            endpoint=create_dataset_status_endpoint(
                max_age=app_config.admin.max_age,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
                hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
            ),
        ),
        Route(
            "/num-dataset-infos-by-builder-name",
            endpoint=create_num_dataset_infos_by_builder_name_endpoint(
                max_age=app_config.admin.max_age,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
                hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
            ),
        ),
        Route(
            "/recreate-dataset",
            endpoint=create_recreate_dataset_endpoint(
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
                hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
                blocked_datasets=app_config.common.blocked_datasets,
                storage_clients=storage_clients,
            ),
            methods=["POST"],
        ),
    ]
    for processing_step in processing_graph.get_processing_steps():
        # beware: here we assume 1-1 mapping between processing steps and cache kinds (and job types)
        # which is currently the case
        cache_kind = processing_step.cache_kind
        job_type = processing_step.job_type
        input_type = processing_step.input_type
        routes.extend(
            [
                Route(
                    f"/force-refresh/{job_type}",
                    endpoint=create_force_refresh_endpoint(
                        input_type=input_type,
                        job_type=job_type,
                        bonus_difficulty_if_dataset_is_big=processing_step.bonus_difficulty_if_dataset_is_big,
                        hf_endpoint=app_config.common.hf_endpoint,
                        hf_token=app_config.common.hf_token,
                        external_auth_url=app_config.admin.external_auth_url,
                        organization=app_config.admin.hf_organization,
                        hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
                        blocked_datasets=app_config.common.blocked_datasets,
                    ),
                    methods=["POST"],
                ),
                Route(
                    f"/cache-reports/{cache_kind}",
                    endpoint=create_cache_reports_endpoint(
                        cache_kind=cache_kind,
                        cache_reports_num_results=app_config.admin.cache_reports_num_results,
                        max_age=app_config.admin.max_age,
                        external_auth_url=app_config.admin.external_auth_url,
                        organization=app_config.admin.hf_organization,
                        hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
                    ),
                ),
                Route(
                    f"/cache-reports-with-content/{cache_kind}",
                    endpoint=create_cache_reports_with_content_endpoint(
                        cache_kind=cache_kind,
                        cache_reports_with_content_num_results=app_config.admin.cache_reports_with_content_num_results,
                        max_age=app_config.admin.max_age,
                        external_auth_url=app_config.admin.external_auth_url,
                        organization=app_config.admin.hf_organization,
                        hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
                    ),
                ),
            ]
        )

    return Starlette(
        routes=[Mount("/admin", routes=routes)],
        middleware=middleware,
        on_shutdown=[resource.release for resource in resources],
    )


def start() -> None:
    uvicorn_config = UvicornConfig.from_env()
    uvicorn.run(
        "app:create_app",
        host=uvicorn_config.hostname,
        port=uvicorn_config.port,
        factory=True,
        workers=uvicorn_config.num_workers,
    )
