import uvicorn  # type: ignore
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from datasets_preview_backend.config import (
    APP_HOSTNAME,
    APP_PORT,
    LOG_LEVEL,
    WEB_CONCURRENCY,
)
from datasets_preview_backend.io.asset import assets_directory, show_asserts_dir
from datasets_preview_backend.io.logger import init_logger
from datasets_preview_backend.io.mongo import connect_cache
from datasets_preview_backend.routes.cache_reports import cache_reports_endpoint
from datasets_preview_backend.routes.cache_stats import cache_stats_endpoint
from datasets_preview_backend.routes.configs import configs_endpoint
from datasets_preview_backend.routes.datasets import datasets_endpoint
from datasets_preview_backend.routes.healthcheck import healthcheck_endpoint
from datasets_preview_backend.routes.infos import infos_endpoint
from datasets_preview_backend.routes.rows import rows_endpoint
from datasets_preview_backend.routes.splits import splits_endpoint
from datasets_preview_backend.routes.validity_status import valid_datasets_endpoint
from datasets_preview_backend.routes.webhook import webhook_endpoint


def create_app() -> Starlette:
    init_logger(log_level=LOG_LEVEL)  # every worker has its own logger
    connect_cache()
    show_asserts_dir()

    middleware = [Middleware(GZipMiddleware)]
    routes = [
        Mount("/assets", app=StaticFiles(directory=assets_directory, check_dir=True), name="assets"),
        Route("/cache", endpoint=cache_stats_endpoint),
        Route("/cache-reports", endpoint=cache_reports_endpoint),
        Route("/configs", endpoint=configs_endpoint),
        Route("/datasets", endpoint=datasets_endpoint),
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/infos", endpoint=infos_endpoint),
        Route("/rows", endpoint=rows_endpoint),
        Route("/splits", endpoint=splits_endpoint),
        Route("/valid", endpoint=valid_datasets_endpoint),
        Route("/webhook", endpoint=webhook_endpoint, methods=["POST"]),
    ]
    return Starlette(routes=routes, middleware=middleware)


def start() -> None:
    uvicorn.run("app:create_app", host=APP_HOSTNAME, port=APP_PORT, factory=True, workers=WEB_CONCURRENCY)
