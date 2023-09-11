# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import asyncio
import contextlib
import random
from dataclasses import dataclass
from typing import Dict, Tuple
from uuid import uuid4


class RandomValueChangedEvent(asyncio.Event):
    """Subclass of asyncio.Event which is able to send a value to the waiter"""

    _random_value: int

    def __init__(self, *, random_value: int) -> None:
        super().__init__()
        self.set_random_value(random_value=random_value)

    def set_random_value(self, *, random_value: int) -> None:
        self._random_value = random_value
        return super().set()

    async def wait_random_value(self) -> int:
        """The caller is responsible to call self.clear() when the event has been handled"""
        await super().wait()
        return self._random_value


@dataclass
class RandomValuePublisher:
    _watchers: Dict[str, RandomValueChangedEvent]
    _value: int

    @property
    def value(self) -> int:
        return self._value

    def _notify_change(self, *, random_value: int) -> None:
        self._value = random_value
        for event in self._watchers.values():
            event.set_random_value(random_value=self._value)

    def _unsubscribe(self, uuid: str) -> None:
        self._watchers.pop(uuid)

    def _subscribe(self) -> Tuple[str, RandomValueChangedEvent]:
        event = RandomValueChangedEvent(random_value=self._value)
        uuid = uuid4().hex
        self._watchers[uuid] = event
        return (uuid, event)


class RandomValueWatcher:
    """
    Utility to watch the value of a randomly generated value (silly example for test purpose).
    """

    _watch_task: asyncio.Task[None]  # <- not sure about the type

    def __init__(self) -> None:
        self._publisher = RandomValuePublisher(_value=42, _watchers={})

    def start_watching(self) -> None:
        self._watch_task = asyncio.create_task(self._watch_loop())

    async def stop_watching(self) -> None:
        self._watch_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._watch_task

    def subscribe(self) -> Tuple[str, RandomValueChangedEvent]:
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

    async def _watch_loop(self) -> None:
        """
        Change the random value every 10ms to 200ms.
        """
        while True:
            random_duration_s = random.randint(10, 200) / 100  # nosec
            await asyncio.sleep(random_duration_s)
            random_value = random.randint(0, 100)  # nosec
            self._publisher._notify_change(random_value=random_value)
