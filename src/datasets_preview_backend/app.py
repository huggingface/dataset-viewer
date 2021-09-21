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


def start() -> None:
    middleware = [get_token_middleware(datasets_enable_private=DATASETS_ENABLE_PRIVATE)]
    app = Starlette(
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
    uvicorn.run(app, host=APP_HOSTNAME, port=APP_PORT)
