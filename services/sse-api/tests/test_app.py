# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import asyncio
import json
from collections.abc import AsyncGenerator
from http import HTTPStatus

import httpx
import pytest
import pytest_asyncio
import uvicorn
from httpx_sse import aconnect_sse
from libcommon.resources import CacheMongoResource
from libcommon.simple_cache import delete_response, upsert_response
from starlette.applications import Starlette

from sse_api.app import create_app_with_config
from sse_api.config import AppConfig
from sse_api.constants import HUB_CACHE_KIND

from .uvicorn_server import UvicornServer


@pytest_asyncio.fixture(scope="function")
async def app_test(
    app_config: AppConfig, event_loop: asyncio.events.AbstractEventLoop
) -> AsyncGenerator[Starlette, None]:
    app = create_app_with_config(app_config)
    config = uvicorn.Config(app=app, port=5555, log_level="warning", loop="asyncio")  # event_loop)

    server = UvicornServer(config)
    await server.start()
    try:
        yield app
    finally:
        await server.shutdown()


@pytest_asyncio.fixture(scope="function")
async def client(app_test: Starlette) -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(base_url=APP_HOST) as client:
        yield client


APP_HOST = "http://localhost:5555"


@pytest.mark.asyncio
async def test_provided_loop_is_running_loop(event_loop: asyncio.events.AbstractEventLoop) -> None:
    assert event_loop is asyncio.get_running_loop()


@pytest.mark.asyncio
async def test_get_healthcheck(client: httpx.AsyncClient) -> None:
    response = await client.get("/healthcheck")
    assert response.status_code == 200
    assert response.text == "ok"


@pytest.mark.asyncio
async def test_metrics(client: httpx.AsyncClient) -> None:
    response = await client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    lines = text.split("\n")
    # examples:
    # starlette_requests_total{method="GET",path_template="/metrics"} 1.0
    # method_steps_processing_time_seconds_sum{method="healthcheck_endpoint",step="all"} 1.6772013623267412e-05
    metrics = {
        parts[0]: float(parts[1]) for line in lines if line and line[0] != "#" and (parts := line.rsplit(" ", 1))
    }

    # the metrics should contain at least the following
    for name in [
        'starlette_requests_total{method="GET",path_template="/metrics"}',
        'method_steps_processing_time_seconds_sum{context="None",method="healthcheck_endpoint",step="all"}',
    ]:
        assert name in metrics, metrics
        assert metrics[name] > 0, metrics


@pytest.mark.asyncio
async def test_hub_cache_only_updates(
    client: httpx.AsyncClient,
    cache_mongo_resource: CacheMongoResource,
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    async def update_hub_cache() -> None:
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset2",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND + "-NOT",
            dataset="dataset2",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": False,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": False,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },  # ^ not important
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        await asyncio.sleep(0.2)
        delete_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": False,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },  # ^ not important
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": False,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },  # ^ not important
            http_status=HTTPStatus.OK,
        )

    update_task = event_loop.create_task(update_hub_cache())

    try:
        async with aconnect_sse(client, f"{APP_HOST}/hub-cache") as event_source:
            event_iter = event_source.aiter_sse()
            for expected_data in [
                {
                    "dataset": "dataset1",
                    "operation": "insert",
                    "hub_cache": {
                        "preview": True,
                        "viewer": True,
                        "partial": False,
                        "num_rows": 100,
                    },
                },
                {
                    "dataset": "dataset2",
                    "operation": "insert",
                    "hub_cache": {
                        "preview": True,
                        "viewer": True,
                        "partial": False,
                        "num_rows": 100,
                    },
                },
                {
                    "dataset": "dataset1",
                    "operation": "update",
                    "hub_cache": {
                        "preview": False,
                        "viewer": True,
                        "partial": False,
                        "num_rows": 100,
                    },
                },
                {"dataset": "dataset1", "operation": "update", "hub_cache": None},
                {"dataset": "dataset1", "operation": "delete", "hub_cache": None},
                {"dataset": "dataset1", "operation": "insert", "hub_cache": None},
                {
                    "dataset": "dataset1",
                    "operation": "update",
                    "hub_cache": {
                        "preview": False,
                        "viewer": True,
                        "partial": False,
                        "num_rows": 100,
                    },
                },
            ]:
                event = await event_iter.__anext__()
                # event = await anext(event_iter)
                # ^ only available in 3.10
                assert event.event == "message", event.data
                assert event.data == json.dumps(expected_data)
    except Exception as err:
        update_task.cancel()
        raise err
    else:
        await update_task


@pytest.mark.asyncio
async def test_hub_cache_only_initialization(
    client: httpx.AsyncClient,
    cache_mongo_resource: CacheMongoResource,
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    # prepare the content of the cache
    upsert_response(
        kind=HUB_CACHE_KIND,
        dataset="dataset1",
        content={
            "preview": True,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=HUB_CACHE_KIND,
        dataset="dataset2",
        content={
            "preview": True,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=HUB_CACHE_KIND + "-NOT",
        dataset="dataset1",
        content={
            "preview": True,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
        http_status=HTTPStatus.OK,
    )
    upsert_response(
        kind=HUB_CACHE_KIND,
        dataset="dataset3",
        content={
            "preview": False,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
        http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
    )

    async with aconnect_sse(client, f"{APP_HOST}/hub-cache") as event_source:
        event_iter = event_source.aiter_sse()
        for expected_data in [
            {
                "dataset": "dataset1",
                "operation": "init",
                "hub_cache": {
                    "preview": True,
                    "viewer": True,
                    "partial": False,
                    "num_rows": 100,
                },
            },
            {
                "dataset": "dataset2",
                "operation": "init",
                "hub_cache": {
                    "preview": True,
                    "viewer": True,
                    "partial": False,
                    "num_rows": 100,
                },
            },
            {"dataset": "dataset3", "operation": "init", "hub_cache": None},
        ]:
            event = await event_iter.__anext__()
            # event = await anext(event_iter)
            # ^ only available in 3.10
            assert event.event == "message", event.data
            assert event.data == json.dumps(expected_data)


@pytest.mark.asyncio
async def test_hub_cache_initialization_and_updates(
    client: httpx.AsyncClient,
    cache_mongo_resource: CacheMongoResource,
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    def prepare_initial_content() -> None:
        # prepare the content of the cache
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset2",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        upsert_response(
            kind=HUB_CACHE_KIND + "-NOT",
            dataset="dataset1",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset3",
            content={
                "preview": False,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )

    async def update_hub_cache() -> None:
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset2",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND + "-NOT",
            dataset="dataset2",
            content={
                "preview": True,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": False,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },
            http_status=HTTPStatus.OK,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": False,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },  # ^ not important
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        await asyncio.sleep(0.2)
        delete_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": False,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },  # ^ not important
            http_status=HTTPStatus.INTERNAL_SERVER_ERROR,
        )
        await asyncio.sleep(0.2)
        upsert_response(
            kind=HUB_CACHE_KIND,
            dataset="dataset1",
            content={
                "preview": False,
                "viewer": True,
                "partial": False,
                "num_rows": 100,
            },  # ^ not important
            http_status=HTTPStatus.OK,
        )

    prepare_initial_content()
    update_task = event_loop.create_task(update_hub_cache())
    # ^ I don't think we are testing concurrency between the loop on the initial content and the loop on the updates
    # We should

    try:
        async with aconnect_sse(client, f"{APP_HOST}/hub-cache") as event_source:
            event_iter = event_source.aiter_sse()
            for expected_data in [
                {
                    "dataset": "dataset1",
                    "operation": "init",
                    "hub_cache": {
                        "preview": True,
                        "viewer": True,
                        "partial": False,
                        "num_rows": 100,
                    },
                },
                {
                    "dataset": "dataset2",
                    "operation": "init",
                    "hub_cache": {
                        "preview": True,
                        "viewer": True,
                        "partial": False,
                        "num_rows": 100,
                    },
                },
                {"dataset": "dataset3", "operation": "init", "hub_cache": None},
                {
                    "dataset": "dataset1",
                    "operation": "update",
                    "hub_cache": {
                        "preview": False,
                        "viewer": True,
                        "partial": False,
                        "num_rows": 100,
                    },
                },
                {"dataset": "dataset1", "operation": "update", "hub_cache": None},
                {"dataset": "dataset1", "operation": "delete", "hub_cache": None},
                {"dataset": "dataset1", "operation": "insert", "hub_cache": None},
                {
                    "dataset": "dataset1",
                    "operation": "update",
                    "hub_cache": {
                        "preview": False,
                        "viewer": True,
                        "partial": False,
                        "num_rows": 100,
                    },
                },
            ]:
                event = await event_iter.__anext__()
                # event = await anext(event_iter)
                # ^ only available in 3.10
                assert event.event == "message", event.data
                assert event.data == json.dumps(expected_data)
    except Exception as err:
        update_task.cancel()
        raise err
    else:
        await update_task
