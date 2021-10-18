import uvicorn  # type: ignore
from starlette.applications import Starlette
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from datasets_preview_backend.assets import assets_directory, show_asserts_dir
from datasets_preview_backend.cache import show_cache_dir  # type: ignore
from datasets_preview_backend.config import (
    APP_HOSTNAME,
    APP_PORT,
    CACHE_PERSIST,
    LOG_LEVEL,
    WEB_CONCURRENCY,
)
from datasets_preview_backend.logger import init_logger
from datasets_preview_backend.routes import (
    CacheReports,
    CacheStats,
    Configs,
    Datasets,
    HealthCheck,
    Infos,
    Rows,
    Splits,
    ValidDatasets,
    WebHook,
)


def create_app() -> Starlette:
    init_logger(log_level=LOG_LEVEL)  # every worker has its own logger
    show_cache_dir()
    show_asserts_dir()

    return Starlette(
        routes=[
            Route("/healthcheck", endpoint=HealthCheck),
            Route("/datasets", endpoint=Datasets),
            Route("/infos", endpoint=Infos),
            Route("/configs", endpoint=Configs),
            Route("/splits", endpoint=Splits),
            Route("/rows", endpoint=Rows),
            Route("/cache", endpoint=CacheStats),
            Route("/valid", endpoint=ValidDatasets),
            Route("/cache-reports", endpoint=CacheReports),
            Route("/webhook", endpoint=WebHook, methods=["POST"]),
            Mount("/assets", app=StaticFiles(directory=assets_directory, check_dir=True), name="assets"),
        ],
    )


def start() -> None:
    # the cache is shared between workers only if CACHE_PERSIST is set to true
    # if not, only one worker is allowed
    web_concurrency = WEB_CONCURRENCY if CACHE_PERSIST else 1
    uvicorn.run("app:create_app", host=APP_HOSTNAME, port=APP_PORT, factory=True, workers=web_concurrency)
