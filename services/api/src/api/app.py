# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List

import uvicorn  # type: ignore
from libcommon.log import init_logging
from libcommon.processing_graph import ProcessingGraph
from libcommon.resources import CacheMongoResource, QueueMongoResource, Resource
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.routing import BaseRoute, Route
from starlette_prometheus import PrometheusMiddleware

from api.config import AppConfig, UvicornConfig
from api.prometheus import Prometheus
from api.routes.healthcheck import healthcheck_endpoint
from api.routes.processing_step import create_processing_step_endpoint
from api.routes.valid import create_is_valid_endpoint, create_valid_endpoint
from api.routes.webhook import create_webhook_endpoint


def create_app() -> Starlette:
    app_config = AppConfig.from_env()

    init_logging(log_level=app_config.common.log_level)
    # ^ set first to have logs as soon as possible

    prometheus = Prometheus()

    processing_graph = ProcessingGraph(app_config.processing_graph.specification)
    processing_steps = list(processing_graph.steps.values())
    processing_steps_required_by_dataset_viewer = processing_graph.get_steps_required_by_dataset_viewer()
    init_processing_steps = processing_graph.get_first_steps()

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
    if cache_resource.check() is False:
        raise RuntimeError("The connection to the cache database could not be established. Exiting.")
    if not queue_resource.check():
        raise RuntimeError("The connection to the queue database could not be established. Exiting.")

    valid: List[BaseRoute] = [
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
                external_auth_url=app_config.api.external_auth_url,
                processing_steps_for_valid=processing_steps_required_by_dataset_viewer,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        )
        # ^ called by https://github.com/huggingface/model-evaluator
    ]
    processing_step_endpoints: List[BaseRoute] = [
        Route(
            processing_step.endpoint,
            endpoint=create_processing_step_endpoint(
                processing_step=processing_step,
                init_processing_steps=init_processing_steps,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
                external_auth_url=app_config.api.external_auth_url,
                max_age_long=app_config.api.max_age_long,
                max_age_short=app_config.api.max_age_short,
            ),
        )
        for processing_step in processing_steps
    ]
    to_protect: List[BaseRoute] = [
        # called by the Hub webhooks
        Route(
            "/webhook",
            endpoint=create_webhook_endpoint(
                init_processing_steps=init_processing_steps,
                hf_endpoint=app_config.common.hf_endpoint,
                hf_token=app_config.common.hf_token,
            ),
            methods=["POST"],
        ),
    ]
    protected: List[BaseRoute] = [
        Route("/healthcheck", endpoint=healthcheck_endpoint),
        # called by Prometheus
        Route("/metrics", endpoint=prometheus.endpoint),
    ]
    routes: List[BaseRoute] = valid + processing_step_endpoints + to_protect + protected

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
