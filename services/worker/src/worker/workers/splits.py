# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libqueue.queue import EmptyQueue

from ..refresh import refresh_splits
from ..worker import Worker

logger = logging.getLogger(__name__)


class SplitsWorker(Worker):
    hf_endpoint: str
    hf_token: Optional[str]

    def __init__(
        self,
        hf_endpoint: str,
        hf_token: Optional[str] = None,
        max_jobs_per_dataset: Optional[int] = None,
        sleep_seconds: Optional[int] = None,
        max_memory_pct: Optional[int] = None,
        max_load_pct: Optional[int] = None,
    ):
        super().__init__(
            max_jobs_per_dataset=max_jobs_per_dataset,
            sleep_seconds=sleep_seconds,
            max_memory_pct=max_memory_pct,
            max_load_pct=max_load_pct,
        )
        self.hf_endpoint = hf_endpoint
        self.hf_token = hf_token

    def process_next_job(self) -> bool:
        logger.debug("try to process a splits/ job")

        try:
            job_id, dataset, *_ = self.queues.splits.start_job()
            logger.debug(f"job assigned: {job_id} for dataset={dataset}")
        except EmptyQueue:
            logger.debug("no job in the queue")
            return False

        try:
            logger.info(f"compute dataset={dataset}")
            success = refresh_splits(
                self.queues, dataset=dataset, hf_endpoint=self.hf_endpoint, hf_token=self.hf_token
            )
        finally:
            self.queues.splits.finish_job(job_id=job_id, success=success)
            result = "success" if success else "error"
            logger.debug(f"job finished with {result}: {job_id} for dataset={dataset}")
        return True
