# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List

import uvicorn  # type: ignore
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import BaseRoute, Route
from starlette_prometheus import PrometheusMiddleware

from api.config import AppConfig, UvicornConfig
from api.prometheus import Prometheus
from api.routes.healthcheck import healthcheck_endpoint
from api.routes.processing_step import create_processing_step_endpoint
from api.routes.valid import create_is_valid_endpoint, create_valid_endpoint
from api.routes.webhook import create_webhook_endpoint
from api.routes.rows import create_rows_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()
    parquet_processing_step = app_config.processing_graph.graph.get_step("/parquet")
    # ^ can raise an exception. We don't catch it here because we want the app to crash if the config is invalid
    prometheus = Prometheus()

    middleware = [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
        ),
        Middleware(GZipMiddleware),
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]
    valid: List[BaseRoute] = [
        Route(
            "/valid",
            endpoint=create_valid_endpoint(
                processing_steps_for_valid=app_config.processing_graph.graph.get_steps_required_by_dataset_viewer(),
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        ),
        Route(
            "/is-valid",
            endpoint=create_is_valid_endpoint(
                external_auth_url=app_config.external_auth_url,
                processing_steps_for_valid=app_config.processing_graph.graph.get_steps_required_by_dataset_viewer(),
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        )
        # ^ called by https://github.com/huggingface/model-evaluator
    ]
    processing_steps: List[BaseRoute] = [
        Route(
            processing_step.endpoint,
            endpoint=create_processing_step_endpoint(
                processing_step=processing_step,
                init_processing_steps=app_config.processing_graph.graph.get_first_steps(),
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                external_auth_url=app_config.external_auth_url,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        )
        for processing_step in list(app_config.processing_graph.graph.steps.values())
    ]
    to_protect: List[BaseRoute] = [
        # called by the Hub webhooks
        Route(
            "/webhook",
            endpoint=create_webhook_endpoint(
                init_processing_steps=app_config.processing_graph.graph.get_first_steps(),
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
            ),
            methods=["POST"],
        ),
    ]
    protected: List[BaseRoute] = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        # called by Prometheus
        Route("/metrics", endpoint=prometheus.endpoint),
    ]
    random_access: List[BaseRoute] = [
        Route(
            "/rows",
            endpoint=create_rows_endpoint(
                parquet_processing_step=parquet_processing_step,
                external_auth_url=app_config.external_auth_url,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        ),
    ]
    routes: List[BaseRoute] = valid + processing_steps + to_protect + protected + random_access
    return Starlette(routes=routes, middleware=middleware)


def start() -> None:
    uvicorn_config = UvicornConfig.from_env()
    uvicorn.run(
        "app:create_app",
        host=uvicorn_config.hostname,
        port=uvicorn_config.port,
        factory=True,
        workers=uvicorn_config.num_workers,
    )
