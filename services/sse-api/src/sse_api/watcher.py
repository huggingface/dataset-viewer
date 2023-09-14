# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import asyncio
import contextlib
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple
from uuid import uuid4

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import PyMongoError

from sse_api.constants import HUB_CACHE_KIND

DatasetHubCacheResponse = Mapping[str, Any]


class ChangeStreamInitError(Exception):
    pass


@dataclass
class HubCacheChangedEventValue:
    dataset: str
    operation: str
    hub_cache: Optional[DatasetHubCacheResponse]
    # ^ None if the dataset has been deleted, or the value is an error response


class HubCacheChangedEvent(asyncio.Event):
    """Subclass of asyncio.Event which is able to send a value to the waiter"""

    _hub_cache_value: Optional[HubCacheChangedEventValue]

    def __init__(self, *, hub_cache_value: Optional[HubCacheChangedEventValue] = None):
        super().__init__()
        self._hub_cache_value = hub_cache_value
        super().set()

    def set_value(self, *, hub_cache_value: Optional[HubCacheChangedEventValue] = None) -> None:
        self._hub_cache_value = hub_cache_value
        return super().set()

    async def wait_value(self) -> Optional[HubCacheChangedEventValue]:
        """The caller is responsible to call self.clear() when the event has been handled"""
        await super().wait()
        return self._hub_cache_value


@dataclass
class HubCachePublisher:
    _watchers: Dict[str, HubCacheChangedEvent]

    def _notify_change(
        self,
        *,
        dataset: str,
        operation: str,
        hub_cache: Optional[DatasetHubCacheResponse],
        suscriber: Optional[str] = None,
    ) -> None:
        hub_cache_value = HubCacheChangedEventValue(dataset=dataset, operation=operation, hub_cache=hub_cache)
        for watcher, event in self._watchers.items():
            if suscriber is None or suscriber == watcher:
                event.set_value(hub_cache_value=hub_cache_value)

    def _unsubscribe(self, uuid: str) -> None:
        self._watchers.pop(uuid)

    def _subscribe(self) -> Tuple[str, HubCacheChangedEvent]:
        event = HubCacheChangedEvent()
        uuid = uuid4().hex
        self._watchers[uuid] = event
        return (uuid, event)


class HubCacheWatcher:
    """
    Utility to watch the value of the cache entries with kind 'dataset-hub-cache'.
    """

    _watch_task: asyncio.Task[None]  # <- not sure about the type

    def __init__(self, client: AsyncIOMotorClient, db_name: str, collection_name: str) -> None:
        self._client = client
        self._collection = self._client[db_name][collection_name]
        self._publisher = HubCachePublisher(_watchers={})

    def run_initialization(self, suscriber: str) -> asyncio.Task[Any]:
        return asyncio.create_task(self._init_loop(suscriber=suscriber))

    def start_watching(self) -> None:
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop_watching(self) -> None:
        self._watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._watch_task

    def subscribe(self) -> Tuple[str, HubCacheChangedEvent]:
        """
        Subscribe to random value changes for the given space.
        The caller is responsible for calling `self.unsubscribe` to release resources.

        Returns:
            (str, RandomValueChangedEvent):
                A 2-tuple containing a UUID and an instance of RandomValueChangedEvent.
                RandomValueChangedEvent can be `await`ed to be notified of updates to the random value.
                UUID must be passed when unsubscribing to release the associated resources.
        """
        return self._publisher._subscribe()

    def unsubscribe(self, uuid: str) -> None:
        """
        Release resources allocated to subscribe to the random value changes.
        """
        pub = self._publisher
        pub._unsubscribe(uuid)

    async def _init_loop(self, suscriber: str) -> None:
        """
        publish an event for each initial dataset-hub-cache cache entry.

        TODO: we don't want to send to all the suscribers
        """

        async for document in self._collection.find(
            filter={"kind": HUB_CACHE_KIND},
            projection={"dataset": 1, "content": 1, "http_status": 1},
            sort=[("_id", 1)],
            batch_size=1,
        ):
            # ^ should we use batch_size=100 instead, and send a list of contents?
            dataset = document["dataset"]
            self._publisher._notify_change(
                suscriber=suscriber,
                dataset=dataset,
                operation="init",
                hub_cache=(document["content"] if document["http_status"] == HTTPStatus.OK else None),
            )

    async def _watch_loop(self) -> None:
        """
        publish a new event, on every change in a dataset-hub-cache cache entry.
        """
        pipeline: Sequence[Mapping[str, Any]] = [
            {
                "$match": {
                    "$or": [
                        {"fullDocument.kind": HUB_CACHE_KIND},
                        {"fullDocumentBeforeChange.kind": HUB_CACHE_KIND},
                    ],
                    "operationType": {"$in": ["insert", "update", "replace", "delete"]},
                },
            },
            {
                "$project": {
                    "fullDocument": 1,
                    "fullDocumentBeforeChange": 1,
                    "updateDescription": 1,
                    "operationType": 1,
                },
            },
        ]
        resume_token = None
        while True:
            try:
                async with self._collection.watch(
                    pipeline,
                    resume_after=resume_token,
                    full_document="updateLookup",
                    full_document_before_change="whenAvailable",
                ) as stream:
                    async for change in stream:
                        resume_token = stream.resume_token
                        operation = change["operationType"]
                        if (
                            operation == "delete"
                            and "fullDocumentBeforeChange" in change
                            and change["fullDocumentBeforeChange"]["kind"] == HUB_CACHE_KIND
                        ):
                            dataset = change["fullDocumentBeforeChange"]["dataset"]
                            self._publisher._notify_change(dataset=dataset, operation=operation, hub_cache=None)
                            continue

                        if change["fullDocument"]["kind"] != HUB_CACHE_KIND:
                            continue

                        if operation == "update" and not any(
                            field in change["updateDescription"]["updatedFields"]
                            for field in ["content", "http_status"]
                        ):
                            # ^ no change, skip
                            continue

                        self._publisher._notify_change(
                            dataset=change["fullDocument"]["dataset"],
                            operation=operation,
                            hub_cache=(
                                change["fullDocument"]["content"]
                                if change["fullDocument"]["http_status"] == HTTPStatus.OK
                                else None
                            ),
                        )
            except PyMongoError:
                if resume_token is None:
                    raise ChangeStreamInitError()
