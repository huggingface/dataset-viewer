# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import uvicorn
from libapi.config import UvicornConfig
from libapi.routes.healthcheck import healthcheck_endpoint
from libapi.routes.metrics import create_metrics_endpoint
from libapi.utils import EXPOSED_HEADERS
from libcommon.cloudfront import get_cloudfront_signer
from libcommon.log import init_logging
from libcommon.resources import CacheMongoResource, QueueMongoResource, Resource
from libcommon.storage_client import StorageClient
from libcommon.url_preparator import URLPreparator
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from webhook.config import AppConfig
from webhook.routes.webhook import create_webhook_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()
    return create_app_with_config(app_config=app_config)


def create_app_with_config(app_config: AppConfig) -> Starlette:
    init_logging(level=app_config.log.level)
    # ^ set first to have logs as soon as possible

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
            allow_credentials=True,
            expose_headers=EXPOSED_HEADERS,
        ),
        Middleware(GZipMiddleware),
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]

    cached_assets_storage_client = StorageClient(
        protocol=app_config.cached_assets.storage_protocol,
        storage_root=app_config.cached_assets.storage_root,
        base_url=app_config.cached_assets.base_url,
        s3_config=app_config.s3,
        # no need to specify a url_signer
    )

    url_signer = get_cloudfront_signer(cloudfront_config=app_config.cloudfront)
    url_preparator = URLPreparator(url_signer=url_signer)
    assets_storage_client = StorageClient(
        protocol=app_config.assets.storage_protocol,
        storage_root=app_config.assets.storage_root,
        base_url=app_config.assets.base_url,
        s3_config=app_config.s3,
        url_preparator=url_preparator,
    )
    storage_clients = [cached_assets_storage_client, assets_storage_client]

    cache_resource = CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url)
    queue_resource = QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url)
    resources: list[Resource] = [cache_resource, queue_resource]
    if not cache_resource.is_available():
        raise RuntimeError("The connection to the cache database could not be established. Exiting.")
    if not queue_resource.is_available():
        raise RuntimeError("The connection to the queue database could not be established. Exiting.")

    routes = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/metrics", endpoint=create_metrics_endpoint()),
        # ^ called by Prometheus
        Route(
            "/webhook",
            endpoint=create_webhook_endpoint(
                hf_webhook_secret=app_config.api.hf_webhook_secret,
                blocked_datasets=app_config.common.blocked_datasets,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                storage_clients=storage_clients,
            ),
            methods=["POST"],
        ),
        # ^ called by the Hub webhooks
    ]

    return Starlette(routes=routes, middleware=middleware, on_shutdown=[resource.release for resource in resources])


def start() -> None:
    uvicorn_config = UvicornConfig.from_env()
    uvicorn.run(
        "app:create_app",
        host=uvicorn_config.hostname,
        port=uvicorn_config.port,
        factory=True,
        workers=uvicorn_config.num_workers,
    )
