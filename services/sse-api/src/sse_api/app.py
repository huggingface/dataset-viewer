# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import asyncio

import uvicorn
from libapi.config import UvicornConfig
from libapi.routes.healthcheck import healthcheck_endpoint
from libapi.routes.metrics import create_metrics_endpoint
from libapi.utils import EXPOSED_HEADERS
from libcommon.constants import CACHE_COLLECTION_RESPONSES
from libcommon.log import init_logging
from libcommon.resources import CacheMongoResource
from libcommon.simple_cache import CachedResponseDocument
from motor.motor_asyncio import AsyncIOMotorClient
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from sse_api.config import AppConfig
from sse_api.routes.hub_cache import create_hub_cache_endpoint
from sse_api.watcher import HubCacheWatcher


def create_app() -> Starlette:
    app_config = AppConfig.from_env()
    return create_app_with_config(app_config=app_config)


def create_app_with_config(app_config: AppConfig) -> Starlette:
    init_logging(level=app_config.log.level)
    # ^ set first to have logs as soon as possible

    # ensure the collection has changeStreamPreAndPostImages enabled (required to report the delete events)
    with CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url) as resource:
        if not resource.is_available():
            raise Exception("MongoDB is not available")
        resource.create_collection(CachedResponseDocument)
        resource.enable_pre_and_post_images(CACHE_COLLECTION_RESPONSES)

    hub_cache_watcher = HubCacheWatcher(
        client=AsyncIOMotorClient(host=app_config.cache.mongo_url, io_loop=asyncio.get_running_loop()),
        db_name=app_config.cache.mongo_database,
        collection_name=CACHE_COLLECTION_RESPONSES,
    )

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
            expose_headers=EXPOSED_HEADERS,
        ),
        # https://github.com/sysid/sse-starlette
        # > Caveat: SSE streaming does not work in combination with GZipMiddleware.
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]

    routes = [
        Route("/hub-cache", endpoint=create_hub_cache_endpoint(hub_cache_watcher=hub_cache_watcher)),
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/metrics", endpoint=create_metrics_endpoint()),
        # ^ called by Prometheus
    ]

    return Starlette(
        routes=routes,
        middleware=middleware,
        on_startup=[hub_cache_watcher.start_watching],
        on_shutdown=[hub_cache_watcher.stop_watching],
    )


def start() -> None:
    uvicorn_config = UvicornConfig.from_env()
    uvicorn.run(
        "app:create_app",
        host=uvicorn_config.hostname,
        port=uvicorn_config.port,
        factory=True,
        workers=uvicorn_config.num_workers,
    )
