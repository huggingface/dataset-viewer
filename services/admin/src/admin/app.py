import uvicorn  # type: ignore
from libcache.cache import connect_to_cache
from libqueue.queue import connect_to_queue
from libutils.logger import init_logger
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from admin.config import (
    APP_HOSTNAME,
    APP_NUM_WORKERS,
    APP_PORT,
    LOG_LEVEL,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
)
from admin.prometheus import Prometheus
from admin.routes.cache_reports import cache_reports_endpoint
from admin.routes.healthcheck import healthcheck_endpoint
from admin.routes.pending_jobs import pending_jobs_endpoint


def create_app() -> Starlette:
    init_logger(log_level=LOG_LEVEL)
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)
    prometheus = Prometheus()

    middleware = [Middleware(GZipMiddleware), Middleware(PrometheusMiddleware, filter_unhandled_paths=True)]
    routes = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/metrics", endpoint=prometheus.endpoint),
        # used by https://observablehq.com/@huggingface/quality-assessment-of-datasets-loading
        Route("/cache-reports", endpoint=cache_reports_endpoint),
        # used in a browser tab to monitor the queue
        Route("/pending-jobs", endpoint=pending_jobs_endpoint),
    ]
    return Starlette(routes=routes, middleware=middleware)


def start() -> None:
    uvicorn.run("app:create_app", host=APP_HOSTNAME, port=APP_PORT, factory=True, workers=APP_NUM_WORKERS)
