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

    def should_skip_job(self, dataset: str, config: Optional[str] = None, split: Optional[str] = None) -> bool:
        return super().should_skip_job(dataset, config, split)

    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> bool:
        return super().compute(dataset, config, split)
