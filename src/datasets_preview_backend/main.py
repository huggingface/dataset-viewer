from starlette.applications import Starlette
from starlette.routing import Route
import uvicorn

from datasets_preview_backend.config import PORT
from datasets_preview_backend.routes import healthcheck, rows, configs, splits, info


def app():
    return Starlette(
        routes=[
            Route("/healthcheck", endpoint=healthcheck),
            Route("/rows", endpoint=rows),
            Route("/configs", endpoint=configs),
            Route("/splits", endpoint=splits),
            Route("/info", endpoint=info),
        ]
    )


if __name__ == "__main__":
    uvicorn.run(app(), host="0.0.0.0", port=PORT)
