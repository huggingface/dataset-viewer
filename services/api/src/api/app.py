# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List

import uvicorn  # type: ignore
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import BaseRoute, Mount, Route
from starlette.staticfiles import StaticFiles
from starlette_prometheus import PrometheusMiddleware

from api.config import AppConfig, UvicornConfig
from api.prometheus import Prometheus
from api.routes.first_rows import create_first_rows_endpoint
from api.routes.healthcheck import healthcheck_endpoint
from api.routes.splits import create_splits_endpoint
from api.routes.valid import create_is_valid_endpoint, valid_endpoint
from api.routes.webhook import create_webhook_endpoint


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
    documented: List[BaseRoute] = [
        Route("/valid", endpoint=valid_endpoint),
        Route(
            "/is-valid",
            endpoint=create_is_valid_endpoint(
                external_auth_url=app_config.api.external_auth_url,
            ),
        ),
        # ^ called by https://github.com/huggingface/model-evaluator
        Route(
            "/first-rows",
            endpoint=create_first_rows_endpoint(
                external_auth_url=app_config.api.external_auth_url,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
            ),
        ),
        Route(
            "/splits",
            endpoint=create_splits_endpoint(
                external_auth_url=app_config.api.external_auth_url,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
            ),
        ),
    ]
    to_protect: List[BaseRoute] = [
        # called by the Hub webhooks
        Route(
            "/webhook",
            endpoint=create_webhook_endpoint(
                hf_endpoint=app_config.common.hf_endpoint, hf_token=app_config.common.hf_token
            ),
            methods=["POST"],
        ),
    ]
    protected: List[BaseRoute] = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        # called by Prometheus
        Route("/metrics", endpoint=prometheus.endpoint),
    ]
    for_development_only: List[BaseRoute] = [
        # it can only be accessed in development. In production the reverse-proxy serves the assets
        Mount(
            "/assets",
            app=StaticFiles(directory=app_config.cache.assets_directory, check_dir=True),
            name="assets",
        ),
    ]
    routes: List[BaseRoute] = documented + to_protect + protected + for_development_only
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
