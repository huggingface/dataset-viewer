# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import uvicorn  # type: ignore
from libcache.simple_cache import connect_to_cache
from libcommon.logger import init_logger
from libqueue.queue import connect_to_queue
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from admin.config import AppConfig, UvicornConfig
from admin.prometheus import Prometheus
from admin.routes.cache_reports import create_cache_reports_endpoint
from admin.routes.healthcheck import healthcheck_endpoint
from admin.routes.pending_jobs import create_pending_jobs_endpoint


def create_app() -> Starlette:
    app_config = AppConfig()
    init_logger(app_config.common.log_level)
    connect_to_cache(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url)
    connect_to_queue(database=app_config.queue.mongo_database, host=app_config.cache.mongo_url)
    prometheus = Prometheus(prometheus_multiproc_dir=app_config.admin.prometheus_multiproc_dir)

    middleware = [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
        ),
        Middleware(GZipMiddleware),
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]
    routes = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/metrics", endpoint=prometheus.endpoint),
        # used by https://observablehq.com/@huggingface/quality-assessment-of-datasets-loading
        Route(
            "/cache-reports/features",
            endpoint=create_cache_reports_endpoint(
                endpoint="features",
                cache_reports_num_results=app_config.admin.cache_reports_num_results,
                max_age=app_config.admin.max_age,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
            ),
        ),
        Route(
            "/cache-reports/first-rows",
            endpoint=create_cache_reports_endpoint(
                endpoint="first-rows",
                cache_reports_num_results=app_config.admin.cache_reports_num_results,
                max_age=app_config.admin.max_age,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
            ),
        ),
        Route(
            "/cache-reports/splits",
            endpoint=create_cache_reports_endpoint(
                endpoint="splits",
                cache_reports_num_results=app_config.admin.cache_reports_num_results,
                max_age=app_config.admin.max_age,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
            ),
        ),
        # used in a browser tab to monitor the queue
        Route(
            "/pending-jobs",
            endpoint=create_pending_jobs_endpoint(
                max_age=app_config.admin.max_age,
                external_auth_url=app_config.admin.external_auth_url,
                organization=app_config.admin.hf_organization,
            ),
        ),
    ]
    return Starlette(routes=routes, middleware=middleware)


def start() -> None:
    uvicorn_config = UvicornConfig()
    uvicorn.run(
        "app:create_app",
        host=uvicorn_config.hostname,
        port=uvicorn_config.port,
        factory=True,
        workers=uvicorn_config.num_workers,
    )
