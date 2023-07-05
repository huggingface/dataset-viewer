# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import uvicorn
from libapi.config import UvicornConfig
from libapi.jwt_token import fetch_jwt_public_key
from libapi.routes.healthcheck import healthcheck_endpoint
from libapi.routes.metrics import create_metrics_endpoint
from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource, Resource
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from api.config import AppConfig, EndpointConfig
from api.routes.endpoint import EndpointsDefinition, create_endpoint
from api.routes.valid import create_valid_endpoint
from api.routes.webhook import create_webhook_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()
    endpoint_config = EndpointConfig.from_env()
    return create_app_with_config(app_config=app_config, endpoint_config=endpoint_config)


def create_app_with_config(app_config: AppConfig, endpoint_config: EndpointConfig) -> Starlette:
    init_logging(level=app_config.log.level)
    # ^ set first to have logs as soon as possible

    processing_graph = ProcessingGraph(app_config.processing_graph.specification)
    endpoints_definition = EndpointsDefinition(processing_graph, endpoint_config)
    hf_jwt_public_key = (
        fetch_jwt_public_key(
            url=app_config.api.hf_jwt_public_key_url,
            hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
            hf_timeout_seconds=app_config.api.hf_timeout_seconds,
        )
        if app_config.api.hf_jwt_public_key_url and app_config.api.hf_jwt_algorithm
        else None
    )

    middleware = [
        Middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
        ),
        Middleware(GZipMiddleware),
        Middleware(PrometheusMiddleware, filter_unhandled_paths=True),
    ]

    cache_resource = CacheMongoResource(database=app_config.cache.mongo_database, host=app_config.cache.mongo_url)
    queue_resource = QueueMongoResource(database=app_config.queue.mongo_database, host=app_config.queue.mongo_url)
    resources: list[Resource] = [cache_resource, queue_resource]
    if not cache_resource.is_available():
        raise RuntimeError("The connection to the cache database could not be established. Exiting.")
    if not queue_resource.is_available():
        raise RuntimeError("The connection to the queue database could not be established. Exiting.")

    routes = [
        Route(
            endpoint_name,
            endpoint=create_endpoint(
                endpoint_name=endpoint_name,
                steps_by_input_type=steps_by_input_type,
                processing_graph=processing_graph,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                hf_jwt_public_key=hf_jwt_public_key,
                hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
                external_auth_url=app_config.api.external_auth_url,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
                cache_max_days=app_config.cache.max_days,
            ),
        )
        for endpoint_name, steps_by_input_type in endpoints_definition.steps_by_input_type_and_endpoint.items()
    ] + [
        Route(
            "/valid",
            endpoint=create_valid_endpoint(
                processing_graph=processing_graph,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        ),
        # ^ called by https://github.com/huggingface/model-evaluator
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/metrics", endpoint=create_metrics_endpoint()),
        # ^ called by Prometheus
        Route(
            "/webhook",
            endpoint=create_webhook_endpoint(
                processing_graph=processing_graph,
                hf_webhook_secret=app_config.api.hf_webhook_secret,
                cache_max_days=app_config.cache.max_days,
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
