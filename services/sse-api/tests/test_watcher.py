# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The HuggingFace Authors.

import asyncio

import pytest
from libcommon.resources import CacheMongoResource
from motor.motor_asyncio import AsyncIOMotorClient

from sse_api.constants import HUB_CACHE_KIND
from sse_api.watcher import HubCacheWatcher

COLLECTION = "test_watcher"


async def wait_for_dataset(watcher: HubCacheWatcher, dataset: str) -> None:
    _, event = watcher.subscribe()
    event.clear()
    while True:
        value = await asyncio.wait_for(event.wait_value(), timeout=10)
        event.clear()
        if value is not None and value.dataset == dataset:
            return


@pytest.mark.asyncio
async def test_the_watcher_follows_the_collection_after_it_is_dropped(
    cache_mongo_resource: CacheMongoResource, cache_mongo_host: str
) -> None:
    client: AsyncIOMotorClient = AsyncIOMotorClient(host=cache_mongo_host)
    database = client[cache_mongo_resource.database]
    await database.create_collection(COLLECTION, changeStreamPreAndPostImages={"enabled": True})
    watcher = HubCacheWatcher(client=client, db_name=cache_mongo_resource.database, collection_name=COLLECTION)
    watcher.start_watching()
    try:
        await asyncio.sleep(0.5)
        document = {"kind": HUB_CACHE_KIND, "content": {}, "http_status": 200}
        await database[COLLECTION].insert_one({**document, "dataset": "before"})
        await wait_for_dataset(watcher, "before")

        # the tests and the cache maintenance drop the collection: the change stream gets an invalidate event
        await database.drop_collection(COLLECTION)
        await database.create_collection(COLLECTION, changeStreamPreAndPostImages={"enabled": True})
        await asyncio.sleep(0.5)
        await database[COLLECTION].insert_one({**document, "dataset": "after"})
        await wait_for_dataset(watcher, "after")
    finally:
        await watcher.stop_watching()
        await database.drop_collection(COLLECTION)
        client.close()
