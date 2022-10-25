from typing import Optional


from libqueue.config import QueueConfig
from libqueue.queue import Queue
from libqueue.worker import Worker


class DummyWorker(Worker):
    def __init__(self, version: str):
        super().__init__(queue_config=QueueConfig(), version=version)

    @property
    def queue(self) -> Queue:
        return Queue("queue_type")

    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> bool:
        pass
