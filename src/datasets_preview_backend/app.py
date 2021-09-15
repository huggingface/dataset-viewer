import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route

from datasets_preview_backend.config import APP_HOSTNAME, APP_PORT
from datasets_preview_backend.routes import configs, healthcheck, info, rows, splits


def start():
    app = Starlette(
        routes=[
            Route("/healthcheck", endpoint=healthcheck),
            Route("/rows", endpoint=rows),
            Route("/configs", endpoint=configs),
            Route("/splits", endpoint=splits),
            Route("/info", endpoint=info),
        ]
    )
    uvicorn.run(app, host=APP_HOSTNAME, port=APP_PORT)
