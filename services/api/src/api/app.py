from typing import List

import uvicorn  # type: ignore
from libcache.asset import init_assets_dir, show_assets_dir
from libcache.cache import connect_to_cache
from libqueue.queue import connect_to_queue
from libutils.logger import init_logger
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import BaseRoute, Mount, Route
from starlette.staticfiles import StaticFiles
from starlette_prometheus import PrometheusMiddleware

from api.config import (
    APP_HOSTNAME,
    APP_NUM_WORKERS,
    APP_PORT,
    ASSETS_DIRECTORY,
    LOG_LEVEL,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
)
from api.prometheus import Prometheus
from api.routes.cache_reports import cache_reports_endpoint
from api.routes.first_rows import first_rows_endpoint
from api.routes.healthcheck import healthcheck_endpoint
from api.routes.pending_jobs import pending_jobs_endpoint
from api.routes.rows import rows_endpoint
from api.routes.splits import splits_endpoint
from api.routes.splits_next import splits_endpoint_next
from api.routes.valid import is_valid_endpoint, valid_datasets_endpoint
from api.routes.webhook import webhook_endpoint


def create_app() -> Starlette:
    init_logger(log_level=LOG_LEVEL)
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)
    show_assets_dir(ASSETS_DIRECTORY)
    prometheus = Prometheus()

    middleware = [Middleware(GZipMiddleware), Middleware(PrometheusMiddleware, filter_unhandled_paths=True)]
    public: List[BaseRoute] = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/valid", endpoint=valid_datasets_endpoint),
        Route("/first-rows", endpoint=first_rows_endpoint),
        Route("/splits-next", endpoint=splits_endpoint_next),
    ]
    public_to_deprecate: List[BaseRoute] = [
        Route("/rows", endpoint=rows_endpoint),
        Route("/splits", endpoint=splits_endpoint),
    ]
    public_undocumented: List[BaseRoute] = [
        # called by the Hub webhooks
        Route("/webhook", endpoint=webhook_endpoint, methods=["POST"]),
        # called by Prometheus
        Route("/metrics", endpoint=prometheus.endpoint),
        # called by https://github.com/huggingface/model-evaluator
        Route("/is-valid", endpoint=is_valid_endpoint),
        # it can be used for development, but in production the reverse-proxy directly serves the assets
        Mount("/assets", app=StaticFiles(directory=init_assets_dir(ASSETS_DIRECTORY), check_dir=True), name="assets"),
    ]
    technical_reports: List[BaseRoute] = [
        # only used by https://observablehq.com/@huggingface/quality-assessment-of-datasets-loading
        Route("/cache-reports", endpoint=cache_reports_endpoint),
        # used in a browser tab to monitor the queue
        Route("/pending-jobs", endpoint=pending_jobs_endpoint),
    ]
    routes: List[BaseRoute] = public + public_to_deprecate + public_undocumented + technical_reports
    return Starlette(routes=routes, middleware=middleware)


def start() -> None:
    uvicorn.run("app:create_app", host=APP_HOSTNAME, port=APP_PORT, factory=True, workers=APP_NUM_WORKERS)
