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

    def should_skip_job(
        self, dataset: str, config: Optional[str] = None, split: Optional[str] = None, force: bool = False
    ) -> bool:
        return super().should_skip_job(dataset=dataset, config=config, split=split, force=force)

    def compute(
        self, dataset: str, config: Optional[str] = None, split: Optional[str] = None, force: bool = False
    ) -> bool:
        return super().compute(dataset=dataset, config=config, split=split, force=force)
