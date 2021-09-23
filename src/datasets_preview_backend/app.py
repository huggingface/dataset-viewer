import uvicorn  # type: ignore
from starlette.applications import Starlette
from starlette.routing import Route

from datasets_preview_backend.config import (
    APP_HOSTNAME,
    APP_PORT,
    DATASETS_ENABLE_PRIVATE,
)
from datasets_preview_backend.middleware.token import get_token_middleware
from datasets_preview_backend.routes import (
    CacheStats,
    Configs,
    Datasets,
    HealthCheck,
    Info,
    Rows,
    Splits,
)


def create_app() -> Starlette:
    middleware = [get_token_middleware(datasets_enable_private=DATASETS_ENABLE_PRIVATE)]
    return Starlette(
        routes=[
            Route("/healthcheck", endpoint=HealthCheck),
            Route("/datasets", endpoint=Datasets),
            Route("/info", endpoint=Info),
            Route("/configs", endpoint=Configs),
            Route("/splits", endpoint=Splits),
            Route("/rows", endpoint=Rows),
            Route("/cache", endpoint=CacheStats),
        ],
        middleware=middleware,
    )


def start() -> None:
    # the cache is not shared between workers for now, thus only one worker is allowed
    WEB_CONCURRENCY = 1
    uvicorn.run("app:create_app", host=APP_HOSTNAME, port=APP_PORT, factory=True, workers=WEB_CONCURRENCY)
