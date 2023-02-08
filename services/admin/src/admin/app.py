# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import uvicorn  # type: ignore
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import (
    AssetsDirectoryResource,
    CacheDatabaseResource,
    LogResource,
    QueueDatabaseResource,
    Resource,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from admin.config import AppConfig, UvicornConfig
from admin.prometheus import Prometheus
from admin.routes.backfill import create_backfill_endpoint
from admin.routes.cache_reports import create_cache_reports_endpoint
from admin.routes.cache_reports_with_content import (
    create_cache_reports_with_content_endpoint,
)
from admin.routes.cancel_jobs import create_cancel_jobs_endpoint
from admin.routes.force_refresh import create_force_refresh_endpoint
from admin.routes.healthcheck import healthcheck_endpoint
from admin.routes.jobs_duration import create_jobs_duration_per_dataset_endpoint
from admin.routes.pending_jobs import create_pending_jobs_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()

    processing_graph = ProcessingGraph(app_config.processing_graph.specification)
    processing_steps = list(processing_graph.steps.values())
    init_processing_steps = processing_graph.get_first_steps()

    assets_directory_resource = AssetsDirectoryResource(init_storage_directory=app_config.assets.storage_directory)
    resources: list[Resource] = [
        LogResource(log_level=app_config.common.log_level),
        # ^ first resource to be acquired, in order to have logs as soon as possible
        assets_directory_resource,
        CacheDatabaseResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url),
        QueueDatabaseResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url),
    ]
    for resource in resources:
        resource.allocate()

    prometheus = Prometheus(
        processing_steps=processing_steps, assets_storage_directory=assets_directory_resource.storage_directory
    )

    middleware = [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
        ),
        Middleware(GZipMiddleware),
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]
    routes = (
        [
            Route("/healthcheck", endpoint=healthcheck_endpoint),
            Route("/metrics", endpoint=prometheus.endpoint),
            # used in a browser tab to monitor the queue
            Route(
                "/pending-jobs",
                endpoint=create_pending_jobs_endpoint(
                    processing_steps=processing_steps,
                    max_age=app_config.admin.max_age,
                    external_auth_url=app_config.admin.external_auth_url,
                    organization=app_config.admin.hf_organization,
                ),
            ),
        ]
        + [
            Route(
                f"/force-refresh{processing_step.endpoint}",
                endpoint=create_force_refresh_endpoint(
                    processing_step=processing_step,
                    hf_endpoint=app_config.common.hf_endpoint,
                    hf_token=app_config.common.hf_token,
                    external_auth_url=app_config.admin.external_auth_url,
                    organization=app_config.admin.hf_organization,
                ),
                methods=["POST"],
            )
            for processing_step in processing_steps
        ]
        + [
            Route(
                "/backfill",
                endpoint=create_backfill_endpoint(
                    init_processing_steps=init_processing_steps,
                    hf_endpoint=app_config.common.hf_endpoint,
                    hf_token=app_config.common.hf_token,
                    external_auth_url=app_config.admin.external_auth_url,
                    organization=app_config.admin.hf_organization,
                ),
                methods=["POST"],
            )
        ]
        + [
            Route(
                f"/cache-reports{processing_step.endpoint}",
                endpoint=create_cache_reports_endpoint(
                    processing_step=processing_step,
                    cache_reports_num_results=app_config.admin.cache_reports_num_results,
                    max_age=app_config.admin.max_age,
                    external_auth_url=app_config.admin.external_auth_url,
                    organization=app_config.admin.hf_organization,
                ),
            )
            for processing_step in processing_steps
        ]
        + [
            Route(
                f"/cache-reports-with-content{processing_step.endpoint}",
                endpoint=create_cache_reports_with_content_endpoint(
                    processing_step=processing_step,
                    cache_reports_with_content_num_results=app_config.admin.cache_reports_with_content_num_results,
                    max_age=app_config.admin.max_age,
                    external_auth_url=app_config.admin.external_auth_url,
                    organization=app_config.admin.hf_organization,
                ),
            )
            for processing_step in processing_steps
        ]
        + [
            Route(
                f"/cancel-jobs{processing_step.endpoint}",
                endpoint=create_cancel_jobs_endpoint(
                    processing_step=processing_step,
                    external_auth_url=app_config.admin.external_auth_url,
                    organization=app_config.admin.hf_organization,
                ),
                methods=["POST"],
            )
            for processing_step in processing_steps
        ]
        + [
            Route(
                f"/jobs-duration-per-dataset{processing_step.endpoint}",
                endpoint=create_jobs_duration_per_dataset_endpoint(
                    processing_step=processing_step,
                    max_age=app_config.admin.max_age,
                    external_auth_url=app_config.admin.external_auth_url,
                    organization=app_config.admin.hf_organization,
                ),
            )
            for processing_step in processing_steps
        ]
    )

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
