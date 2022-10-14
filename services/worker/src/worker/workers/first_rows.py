# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libqueue.queue import EmptyQueue

from ..refresh import refresh_first_rows
from ..worker import Worker

logger = logging.getLogger(__name__)


class FirstRowsWorker(Worker):
    assets_base_url: str
    hf_endpoint: str
    hf_token: Optional[str]
    max_size_fallback: Optional[int]
    rows_max_bytes: Optional[int]
    rows_max_number: Optional[int]
    rows_min_number: Optional[int]

    def __init__(
        self,
        assets_base_url: str,
        hf_endpoint: str,
        hf_token: Optional[str] = None,
        max_size_fallback: Optional[int] = None,
        rows_max_bytes: Optional[int] = None,
        rows_max_number: Optional[int] = None,
        rows_min_number: Optional[int] = None,
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
        self.assets_base_url = assets_base_url
        self.hf_endpoint = hf_endpoint
        self.hf_token = hf_token
        self.max_size_fallback = max_size_fallback
        self.rows_max_bytes = rows_max_bytes
        self.rows_max_number = rows_max_number
        self.rows_min_number = rows_min_number

    def process_next_job(self) -> bool:
        logger.debug("try to process a first-rows job")

        try:
            job_id, dataset, config, split = self.queues.first_rows.start_job()
            logger.debug(f"job assigned: {job_id} for dataset={dataset} config={config} split={split}")
        except EmptyQueue:
            logger.debug("no job in the queue")
            return False

        success = False
        try:
            logger.info(f"compute dataset={dataset} config={config} split={split}")
            if config is None or split is None:
                raise ValueError("config and split are required")
            http_status = refresh_first_rows(
                dataset=dataset,
                config=config,
                split=split,
                assets_base_url=self.assets_base_url,
                hf_endpoint=self.hf_endpoint,
                hf_token=self.hf_token,
                max_size_fallback=self.max_size_fallback,
                rows_max_bytes=self.rows_max_bytes,
                rows_max_number=self.rows_max_number,
                rows_min_number=self.rows_min_number,
            )
            success = http_status == HTTPStatus.OK
        finally:
            self.queues.first_rows.finish_job(job_id=job_id, success=success)
            result = "success" if success else "error"
            logger.debug(f"job finished with {result}: {job_id} for dataset={dataset} config={config} split={split}")
        return True
