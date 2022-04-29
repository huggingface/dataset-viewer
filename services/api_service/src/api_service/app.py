import uvicorn  # type: ignore
from libcache.asset import init_assets_dir, show_assets_dir
from libcache.cache import connect_to_cache
from libqueue.queue import connect_to_queue
from libutils.logger import init_logger
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from api_service.config import (
    APP_HOSTNAME,
    APP_PORT,
    ASSETS_DIRECTORY,
    LOG_LEVEL,
    MONGO_CACHE_DATABASE,
    MONGO_QUEUE_DATABASE,
    MONGO_URL,
    WEB_CONCURRENCY,
)
from api_service.routes.cache_reports import cache_reports_endpoint
from api_service.routes.cache_stats import cache_stats_endpoint
from api_service.routes.healthcheck import healthcheck_endpoint
from api_service.routes.hf_datasets import (
    hf_datasets_count_by_cache_status_endpoint,
    hf_datasets_endpoint,
)
from api_service.routes.queue_dump import (
    queue_dump_endpoint,
    queue_dump_waiting_started_endpoint,
)
from api_service.routes.queue_stats import queue_stats_endpoint
from api_service.routes.refresh_split import refresh_split_endpoint
from api_service.routes.rows import rows_endpoint
from api_service.routes.splits import splits_endpoint
from api_service.routes.valid import is_valid_endpoint, valid_datasets_endpoint
from api_service.routes.webhook import webhook_endpoint


def create_app() -> Starlette:
    init_logger(log_level=LOG_LEVEL)
    connect_to_cache(database=MONGO_CACHE_DATABASE, host=MONGO_URL)
    connect_to_queue(database=MONGO_QUEUE_DATABASE, host=MONGO_URL)
    show_assets_dir(ASSETS_DIRECTORY)

    middleware = [Middleware(GZipMiddleware)]
    routes = [
        Mount("/assets", app=StaticFiles(directory=init_assets_dir(ASSETS_DIRECTORY), check_dir=True), name="assets"),
        Route("/cache", endpoint=cache_stats_endpoint),
        Route("/cache-reports", endpoint=cache_reports_endpoint),
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/hf_datasets", endpoint=hf_datasets_endpoint),
        Route("/hf-datasets-count-by-cache-status", endpoint=hf_datasets_count_by_cache_status_endpoint),
        Route("/is-valid", endpoint=is_valid_endpoint),
        Route("/queue", endpoint=queue_stats_endpoint),
        Route("/queue-dump-waiting-started", endpoint=queue_dump_waiting_started_endpoint),
        Route("/queue-dump", endpoint=queue_dump_endpoint),
        Route("/refresh-split", endpoint=refresh_split_endpoint, methods=["POST"]),
        Route("/rows", endpoint=rows_endpoint),
        Route("/splits", endpoint=splits_endpoint),
        Route("/valid", endpoint=valid_datasets_endpoint),
        Route("/webhook", endpoint=webhook_endpoint, methods=["POST"]),
    ]
    return Starlette(routes=routes, middleware=middleware)


def start() -> None:
    uvicorn.run("app:create_app", host=APP_HOSTNAME, port=APP_PORT, factory=True, workers=WEB_CONCURRENCY)
