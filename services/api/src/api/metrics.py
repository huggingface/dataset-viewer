import uvicorn  # type: ignore
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import metrics

from api.config import METRICS_HOSTNAME, METRICS_NUM_WORKERS, METRICS_PORT


def create_app() -> Starlette:
    middleware = [Middleware(GZipMiddleware)]
    routes = [Route("/", endpoint=metrics)]
    return Starlette(routes=routes, middleware=middleware)


def start_metrics() -> None:
    uvicorn.run(
        "metrics:create_app", host=METRICS_HOSTNAME, port=METRICS_PORT, factory=True, workers=METRICS_NUM_WORKERS
    )
