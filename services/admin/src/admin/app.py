# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import uvicorn  # type: ignore
from libcommon.processing_steps import PROCESSING_STEPS
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from admin.config import AppConfig, UvicornConfig
from admin.prometheus import Prometheus
from admin.routes.cache_reports import create_cache_reports_endpoint
from admin.routes.cancel_jobs import create_cancel_jobs_endpoint
from admin.routes.force_refresh import create_force_refresh_endpoint
from admin.routes.healthcheck import healthcheck_endpoint
from admin.routes.pending_jobs import create_pending_jobs_endpoint


def create_app() -> Starlette:
    app_config = AppConfig()
    prometheus = Prometheus()

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
                    processing_steps=PROCESSING_STEPS,
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
            for processing_step in PROCESSING_STEPS
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
            for processing_step in PROCESSING_STEPS
        ]
        + [
            Route(
                f"/cancel-jobs{processing_step.endpoint}",
                endpoint=create_cancel_jobs_endpoint(
                    processing_step=processing_step,
                    external_auth_url=app_config.admin.external_auth_url,
                    organization=app_config.admin.hf_organization,
                ),
            )
            for processing_step in PROCESSING_STEPS
        ]
    )
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
