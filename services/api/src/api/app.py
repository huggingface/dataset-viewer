from typing import List

import uvicorn  # type: ignore
from libcache.asset import init_assets_dir, show_assets_dir
from libcache.simple_cache import connect_to_cache
from libqueue.queue import connect_to_queue
from libutils.logger import init_logger
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import BaseRoute, Mount, Route
from starlette.staticfiles import StaticFiles
from starlette_prometheus import PrometheusMiddleware

from api.config import (
    APP_HOSTNAME,
    APP_NUM_WORKERS,
    APP_PORT,
    ASSETS_DIRECTORY,
    EXTERNAL_AUTH_URL,
    LOG_LEVEL,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
)
from api.prometheus import Prometheus
from api.routes.first_rows import create_first_rows_endpoint
from api.routes.healthcheck import healthcheck_endpoint
from api.routes.splits import create_splits_endpoint
from api.routes.valid import create_is_valid_endpoint, valid_endpoint
from api.routes.webhook import webhook_endpoint


def create_app() -> Starlette:
    init_logger(log_level=LOG_LEVEL)
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)
    show_assets_dir(ASSETS_DIRECTORY)
    prometheus = Prometheus()

    middleware = [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
        ),
        Middleware(GZipMiddleware),
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]
    documented: List[BaseRoute] = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/valid", endpoint=valid_endpoint),
        Route("/is-valid", endpoint=create_is_valid_endpoint(EXTERNAL_AUTH_URL)),
        # ^ called by https://github.com/huggingface/model-evaluator
        Route("/first-rows", endpoint=create_first_rows_endpoint(EXTERNAL_AUTH_URL)),
        Route("/splits", endpoint=create_splits_endpoint(EXTERNAL_AUTH_URL)),
    ]
    to_deprecate: List[BaseRoute] = [
        Route("/valid-next", endpoint=valid_endpoint),
        Route("/is-valid-next", endpoint=create_is_valid_endpoint(EXTERNAL_AUTH_URL)),
        Route("/splits-next", endpoint=create_splits_endpoint(EXTERNAL_AUTH_URL)),
    ]
    to_protect: List[BaseRoute] = [
        # called by the Hub webhooks
        Route("/webhook", endpoint=webhook_endpoint, methods=["POST"]),
        # called by Prometheus
        Route("/metrics", endpoint=prometheus.endpoint),
    ]
    for_development_only: List[BaseRoute] = [
        # it can only be accessed in development. In production the reverse-proxy serves the assets
        Mount("/assets", app=StaticFiles(directory=init_assets_dir(ASSETS_DIRECTORY), check_dir=True), name="assets"),
    ]
    routes: List[BaseRoute] = documented + to_deprecate + to_protect + for_development_only
    return Starlette(routes=routes, middleware=middleware)


def start() -> None:
    uvicorn.run("app:create_app", host=APP_HOSTNAME, port=APP_PORT, factory=True, workers=APP_NUM_WORKERS)
