import uvicorn  # type: ignore
from starlette.applications import Starlette
from starlette.routing import Route

from datasets_preview_backend.config import APP_HOSTNAME, APP_PORT
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
    # the cache is not shared between workers for now, thus only one worker is allowed
    WEB_CONCURRENCY = 1
    uvicorn.run("app:create_app", host=APP_HOSTNAME, port=APP_PORT, factory=True, workers=WEB_CONCURRENCY)
