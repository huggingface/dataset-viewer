# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libcache.simple_cache import upsert_first_rows_response
from libqueue.worker import Worker

from .response import get_first_rows_response
from .utils import (
    ConfigNotFoundError,
    DatasetNotFoundError,
    Queues,
    SplitNotFoundError,
    UnexpectedError,
    WorkerCustomError,
)

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
            sleep_seconds=sleep_seconds,
            max_memory_pct=max_memory_pct,
            max_load_pct=max_load_pct,
        )
        self._queues = Queues(max_jobs_per_dataset=max_jobs_per_dataset)
        self.assets_base_url = assets_base_url
        self.hf_endpoint = hf_endpoint
        self.hf_token = hf_token
        self.max_size_fallback = max_size_fallback
        self.rows_max_bytes = rows_max_bytes
        self.rows_max_number = rows_max_number
        self.rows_min_number = rows_min_number

    @property
    def queue(self):
        return self._queues.first_rows

    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> bool:
        if config is None or split is None:
            raise ValueError("config and split are required")
        try:
            response = get_first_rows_response(
                dataset,
                config,
                split,
                assets_base_url=self.assets_base_url,
                hf_endpoint=self.hf_endpoint,
                hf_token=self.hf_token,
                max_size_fallback=self.max_size_fallback,
                rows_max_bytes=self.rows_max_bytes,
                rows_max_number=self.rows_max_number,
                rows_min_number=self.rows_min_number,
            )
            upsert_first_rows_response(dataset, config, split, dict(response), HTTPStatus.OK)
            logger.debug(f"dataset={dataset} config={config} split={split} is valid, cache updated")
            return True
        except (DatasetNotFoundError, ConfigNotFoundError, SplitNotFoundError):
            logger.debug(
                f"the dataset={dataset}, config {config} or split {split} could not be found, don't update the cache"
            )
            return False
        except WorkerCustomError as err:
            upsert_first_rows_response(
                dataset,
                config,
                split,
                dict(err.as_response()),
                err.status_code,
                err.code,
                dict(err.as_response_with_cause()),
            )
            logger.debug(
                f"first-rows response for dataset={dataset} config={config} split={split} had an error, cache updated"
            )
            return False
        except Exception as err:
            e = UnexpectedError(str(err), err)
            upsert_first_rows_response(
                dataset,
                config,
                split,
                dict(e.as_response()),
                e.status_code,
                e.code,
                dict(e.as_response_with_cause()),
            )
            logger.debug(
                f"first-rows response for dataset={dataset} config={config} split={split} had a server"
                " error, cache updated"
            )
            return False
