# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List

import uvicorn  # type: ignore
from libcache.asset import init_assets_dir, show_assets_dir
from libcache.simple_cache import connect_to_cache
from libcommon.logger import init_logger
from libqueue.queue import connect_to_queue
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
    init_logger(app_config.common.log_level)
    connect_to_cache(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url)
    connect_to_queue(database=app_config.queue.mongo_database, host=app_config.cache.mongo_url)
    show_assets_dir(assets_directory=app_config.cache.assets_directory)
    prometheus = Prometheus(prometheus_multiproc_dir=app_config.api.prometheus_multiproc_dir)

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
            app=StaticFiles(
                directory=init_assets_dir(assets_directory=app_config.cache.assets_directory), check_dir=True
            ),
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
