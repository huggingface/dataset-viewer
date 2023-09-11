# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import uvicorn
from libapi.config import UvicornConfig
from libapi.routes.healthcheck import healthcheck_endpoint
from libapi.routes.metrics import create_metrics_endpoint
from libapi.utils import EXPOSED_HEADERS
from libcommon.log import init_logging
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from sse_api.config import AppConfig
from sse_api.routes.numbers import create_numbers_endpoint
from sse_api.watcher import RandomValueWatcher


def create_app() -> Starlette:
    app_config = AppConfig.from_env()
    return create_app_with_config(app_config=app_config)


def create_app_with_config(app_config: AppConfig) -> Starlette:
    init_logging(level=app_config.log.level)
    # ^ set first to have logs as soon as possible

    random_value_watcher = RandomValueWatcher()
    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
            expose_headers=EXPOSED_HEADERS,
        ),
        # https://github.com/sysid/sse-starlette
        # > Caveat: SSE streaming does not work in combination with GZipMiddleware.
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]

    routes = [
        Route("/numbers", endpoint=create_numbers_endpoint(random_value_watcher=random_value_watcher)),
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/metrics", endpoint=create_metrics_endpoint()),
        # ^ called by Prometheus
    ]

    return Starlette(
        routes=routes,
        middleware=middleware,
        on_startup=[random_value_watcher.start_watching],
        on_shutdown=[random_value_watcher.stop_watching],
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
