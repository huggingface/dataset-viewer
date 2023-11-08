# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route

from worker.config import UvicornConfig
from worker.routes.healthcheck import healthcheck_endpoint
from worker.routes.metrics import create_metrics_endpoint


def create_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/healthcheck", endpoint=healthcheck_endpoint),
            Route("/metrics", endpoint=create_metrics_endpoint()),
        ]
    )


if __name__ == "__main__":
    uvicorn_config = UvicornConfig.from_env()
    uvicorn.run(
        "worker.start_web_app:create_app",
        host=uvicorn_config.hostname,
        port=uvicorn_config.port,
        factory=True,
        workers=uvicorn_config.num_workers,
    )
