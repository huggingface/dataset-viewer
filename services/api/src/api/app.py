# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import uvicorn
from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource, Resource
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import Route
from starlette_prometheus import PrometheusMiddleware

from api.config import AppConfig, EndpointConfig, UvicornConfig
from api.jwt_token import fetch_jwt_public_key
from api.prometheus import Prometheus
from api.routes.endpoint import EndpointsDefinition, create_endpoint
from api.routes.healthcheck import healthcheck_endpoint
from api.routes.rows import create_rows_endpoint
from api.routes.valid import create_is_valid_endpoint, create_valid_endpoint
from api.routes.webhook import create_webhook_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()
    endpoint_config = EndpointConfig.from_env()
    return create_app_with_config(app_config=app_config, endpoint_config=endpoint_config)


def create_app_with_config(app_config: AppConfig, endpoint_config: EndpointConfig) -> Starlette:
    init_logging(log_level=app_config.common.log_level)
    # ^ set first to have logs as soon as possible

    prometheus = Prometheus()

    processing_graph = ProcessingGraph(app_config.processing_graph.specification)
    endpoints_definition = EndpointsDefinition(processing_graph, endpoint_config)
    processing_steps_required_by_dataset_viewer = processing_graph.get_steps_required_by_dataset_viewer()
    init_processing_steps = processing_graph.get_first_steps()
    hf_jwt_public_key = (
        fetch_jwt_public_key(
            url=app_config.api.hf_jwt_public_key_url,
            hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
            hf_timeout_seconds=app_config.api.hf_timeout_seconds,
        )
        if app_config.api.hf_jwt_public_key_url and app_config.api.hf_jwt_algorithm
        else None
    )
    parquet_processing_step = processing_graph.get_step("/parquet")
    # ^ can raise an exception. We don't catch it here because we want the app to crash if the config is invalid

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
                init_processing_steps=init_processing_steps,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                hf_jwt_public_key=hf_jwt_public_key,
                hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
                external_auth_url=app_config.api.external_auth_url,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        )
        for endpoint_name, steps_by_input_type in endpoints_definition.steps_by_input_type_and_endpoint.items()
    ] + [
        Route(
            "/valid",
            endpoint=create_valid_endpoint(
                processing_steps_for_valid=processing_steps_required_by_dataset_viewer,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        ),
        Route(
            "/is-valid",
            endpoint=create_is_valid_endpoint(
                hf_jwt_public_key=hf_jwt_public_key,
                hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
                external_auth_url=app_config.api.external_auth_url,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                processing_steps_for_valid=processing_steps_required_by_dataset_viewer,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        ),
        # ^ called by https://github.com/huggingface/model-evaluator
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        Route("/metrics", endpoint=prometheus.endpoint),
        # ^ called by Prometheus
        Route(
            "/webhook",
            endpoint=create_webhook_endpoint(
                init_processing_steps=init_processing_steps,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                hf_webhook_secret=app_config.api.hf_webhook_secret,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
            ),
            methods=["POST"],
        ),
        # ^ called by the Hub webhooks
        Route(
            "/rows",
            endpoint=create_rows_endpoint(
                parquet_processing_step=parquet_processing_step,
                hf_jwt_public_key=hf_jwt_public_key,
                hf_jwt_algorithm=app_config.api.hf_jwt_algorithm,
                external_auth_url=app_config.api.external_auth_url,
                hf_timeout_seconds=app_config.api.hf_timeout_seconds,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        ),
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
