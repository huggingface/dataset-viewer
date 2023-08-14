# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import uvicorn
from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource, Resource
from libcommon.storage import (
    exists,
    init_assets_dir,
    init_duckdb_index_cache_dir,
    init_hf_datasets_cache_dir,
    init_parquet_metadata_dir,
    init_statistics_cache_dir,
)
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from admin.config import AppConfig, UvicornConfig
from admin.routes.cache_reports import create_cache_reports_endpoint
from admin.routes.cache_reports_with_content import (
    create_cache_reports_with_content_endpoint,
)
from admin.routes.dataset_backfill import create_dataset_backfill_endpoint
from admin.routes.dataset_backfill_plan import create_dataset_backfill_plan_endpoint
from admin.routes.dataset_status import create_dataset_status_endpoint
from admin.routes.force_refresh import create_force_refresh_endpoint
from admin.routes.healthcheck import healthcheck_endpoint
from admin.routes.metrics import create_metrics_endpoint
from admin.routes.pending_jobs import create_pending_jobs_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()

    init_logging(level=app_config.log.level)
    # ^ set first to have logs as soon as possible
    assets_directory = init_assets_dir(directory=app_config.assets.storage_directory)
    duckdb_index_cache_directory = init_duckdb_index_cache_dir(directory=app_config.duckdb_index.cache_directory)
    hf_datasets_cache_directory = init_hf_datasets_cache_dir(app_config.datasets_based.hf_datasets_cache)
    parquet_metadata_directory = init_parquet_metadata_dir(directory=app_config.parquet_metadata.storage_directory)
    statistics_cache_directory = init_statistics_cache_dir(app_config.descriptive_statistics.cache_directory)

    if not exists(assets_directory):
        raise RuntimeError("The assets storage directory could not be accessed. Exiting.")

    processing_graph = ProcessingGraph(app_config.processing_graph.specification)

    cache_resource = CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url)
    queue_resource = QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url)
    resources: list[Resource] = [cache_resource, queue_resource]
    if not cache_resource.is_available():
        raise RuntimeError("The connection to the cache database could not be established. Exiting.")
    if not queue_resource.is_available():
        raise RuntimeError("The connection to the queue database could not be established. Exiting.")

    middleware = [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
        ),
        Middleware(GZipMiddleware),
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]
    routes = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route(
            "/metrics",
            endpoint=create_metrics_endpoint(
                assets_directory=assets_directory,
                descriptive_statistics_directory=statistics_cache_directory,
                duckdb_directory=duckdb_index_cache_directory,
                hf_datasets_directory=hf_datasets_cache_directory,
                parquet_metadata_directory=parquet_metadata_directory,
            ),
        ),
        # used in a browser tab to monitor the queue
        Route(
            "/pending-jobs",
            endpoint=create_pending_jobs_endpoint(
                processing_graph=processing_graph,
                max_age=app_config.admin.max_age,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
                hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
            ),
        ),
        Route(
            "/dataset-backfill",
            endpoint=create_dataset_backfill_endpoint(
                processing_graph=processing_graph,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                cache_max_days=app_config.cache.max_days,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
                hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
            ),
            methods=["POST"],
        ),
        Route(
            "/dataset-backfill-plan",
            endpoint=create_dataset_backfill_plan_endpoint(
                processing_graph=processing_graph,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                cache_max_days=app_config.cache.max_days,
                max_age=app_config.admin.max_age,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
                hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
            ),
        ),
        Route(
            "/dataset-status",
            endpoint=create_dataset_status_endpoint(
                processing_graph=processing_graph,
                max_age=app_config.admin.max_age,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
                hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
            ),
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
                        difficulty=processing_step.difficulty,
                        hf_endpoint=app_config.common.hf_endpoint,
                        hf_token=app_config.common.hf_token,
                        external_auth_url=app_config.admin.external_auth_url,
                        organization=app_config.admin.hf_organization,
                        hf_timeout_seconds=app_config.admin.hf_timeout_seconds,
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
