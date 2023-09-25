# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import asyncio
import json
from collections.abc import AsyncGenerator
from http import HTTPStatus
from typing import Any

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


TIMEOUT = 0.5


@pytest_asyncio.fixture(scope="function")
async def client(app_test: Starlette) -> AsyncGenerator[httpx.AsyncClient, None]:
    async with httpx.AsyncClient(base_url=APP_HOST, timeout=TIMEOUT) as client:
        yield client


async def sleep() -> None:
    await asyncio.sleep(TIMEOUT / 10)


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


def init_hub_cache() -> None:
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
    await sleep()
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
    await sleep()
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
    await sleep()
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
    await sleep()
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
    await sleep()
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
    await sleep()
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
    await sleep()
    delete_response(
        kind=HUB_CACHE_KIND,
        dataset="dataset1",
    )
    await sleep()
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
    await sleep()
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


EventsList = list[dict[str, Any]]
INIT_ONLY_EVENTS: EventsList = [
    {
        "dataset": "dataset1",
        "hub_cache": {
            "preview": True,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
    },
    {
        "dataset": "dataset2",
        "hub_cache": {
            "preview": True,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
    },
    {"dataset": "dataset3", "hub_cache": None},
]
UPDATE_ONLY_EVENTS: EventsList = [
    {
        "dataset": "dataset1",
        "hub_cache": {
            "preview": True,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
    },
    {
        "dataset": "dataset2",
        "hub_cache": {
            "preview": True,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
    },
    {
        "dataset": "dataset1",
        "hub_cache": {
            "preview": False,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
    },
    {"dataset": "dataset1", "hub_cache": None},
    {"dataset": "dataset1", "hub_cache": None},
    {"dataset": "dataset1", "hub_cache": None},
    {
        "dataset": "dataset1",
        "hub_cache": {
            "preview": False,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
    },
]
UPDATE_ONLY_AFTER_INIT_EVENTS: EventsList = [
    {
        "dataset": "dataset1",
        "hub_cache": {
            "preview": False,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
    },
    {"dataset": "dataset1", "hub_cache": None},
    {"dataset": "dataset1", "hub_cache": None},
    {"dataset": "dataset1", "hub_cache": None},
    {
        "dataset": "dataset1",
        "hub_cache": {
            "preview": False,
            "viewer": True,
            "partial": False,
            "num_rows": 100,
        },
    },
]
INIT_AND_UPDATE_EVENTS: EventsList = INIT_ONLY_EVENTS + UPDATE_ONLY_AFTER_INIT_EVENTS


async def check(client: httpx.AsyncClient, url: str, expected_events: EventsList) -> None:
    async with aconnect_sse(client, url) as event_source:
        event_iter = event_source.aiter_sse()
        i = 0
        while True:
            try:
                event = await event_iter.__anext__()
                # event = await anext(event_iter)
                # ^ only available in 3.10
                assert event.event == "message", event.data
                assert event.data == json.dumps(expected_events[i])
                i += 1
            except httpx.ReadTimeout:
                break
    assert i == len(expected_events)


@pytest.mark.asyncio
async def test_hub_cache_only_updates(
    client: httpx.AsyncClient,
    cache_mongo_resource: CacheMongoResource,
    event_loop: asyncio.AbstractEventLoop,
) -> None:
    update_task = event_loop.create_task(update_hub_cache())

    try:
        await check(client, f"{APP_HOST}/hub-cache", UPDATE_ONLY_EVENTS)
    except Exception as err:
        update_task.cancel()
        raise err
    else:
        await update_task


@pytest.mark.parametrize(
    ("all", "expected_events"),
    [
        ("?all=true", INIT_ONLY_EVENTS),
        ("", []),
        ("?all=false", []),
    ],
)
@pytest.mark.asyncio
async def test_hub_cache_only_initialization(
    client: httpx.AsyncClient,
    cache_mongo_resource: CacheMongoResource,
    event_loop: asyncio.AbstractEventLoop,
    all: str,
    expected_events: EventsList,
) -> None:
    init_hub_cache()
    await check(client, f"{APP_HOST}/hub-cache{all}", expected_events)


@pytest.mark.parametrize(
    ("all", "expected_events"),
    [
        ("?all=true", INIT_AND_UPDATE_EVENTS),
        ("?all=false", UPDATE_ONLY_AFTER_INIT_EVENTS),
        ("", UPDATE_ONLY_AFTER_INIT_EVENTS),
    ],
)
@pytest.mark.asyncio
async def test_hub_cache_initialization_and_updates(
    client: httpx.AsyncClient,
    cache_mongo_resource: CacheMongoResource,
    event_loop: asyncio.AbstractEventLoop,
    all: str,
    expected_events: EventsList,
) -> None:
    init_hub_cache()
    update_task = event_loop.create_task(update_hub_cache())
    # ^ We are not testing concurrency between the loop on the initial content and the loop on the updates
    try:
        await check(client, f"{APP_HOST}/hub-cache{all}", expected_events)
    except Exception as err:
        update_task.cancel()
        raise err
    else:
        await update_task
