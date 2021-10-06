import uvicorn  # type: ignore
from starlette.applications import Starlette
from starlette.routing import Route

from datasets_preview_backend.config import APP_HOSTNAME, APP_PORT, CACHE_PERSIST, WEB_CONCURRENCY
from datasets_preview_backend.routes import (
    CacheStats,
    Configs,
    Datasets,
    HealthCheck,
    Infos,
    Rows,
    Splits,
    ValidDatasets,
)


def create_app() -> Starlette:
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
        ],
    )


def start() -> None:
    # the cache is shared between workers only if CACHE_PERSIST is set to true
    # if not, only one worker is allowed
    web_concurrency = WEB_CONCURRENCY if CACHE_PERSIST else 1
    uvicorn.run("app:create_app", host=APP_HOSTNAME, port=APP_PORT, factory=True, workers=web_concurrency)
