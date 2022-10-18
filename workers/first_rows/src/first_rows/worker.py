# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libcache.simple_cache import upsert_first_rows_response
from libqueue.worker import Worker

from first_rows.config import WorkerConfig
from first_rows.response import get_first_rows_response
from first_rows.utils import (
    ConfigNotFoundError,
    DatasetNotFoundError,
    Queues,
    SplitNotFoundError,
    UnexpectedError,
    WorkerCustomError,
)

logger = logging.getLogger(__name__)


class FirstRowsWorker(Worker):
    config: WorkerConfig

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(queue_config=worker_config.queue)
        self._queues = Queues(max_jobs_per_dataset=worker_config.queue.max_jobs_per_dataset)
        self.config = worker_config

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
                dataset=dataset,
                config=config,
                split=split,
                assets_base_url=self.config.common.assets_base_url,
                hf_endpoint=self.config.common.hf_endpoint,
                hf_token=self.config.common.hf_token,
                min_cell_bytes=self.config.first_rows.min_cell_bytes,
                max_size_fallback=self.config.first_rows.fallback_max_dataset_size,
                rows_max_bytes=self.config.first_rows.max_bytes,
                rows_max_number=self.config.first_rows.max_number,
                rows_min_number=self.config.first_rows.min_number,
                assets_directory=self.config.cache.assets_directory,
            )
            upsert_first_rows_response(
                dataset_name=dataset,
                config_name=config,
                split_name=split,
                response=dict(response),
                http_status=HTTPStatus.OK,
            )
            logger.debug(f"dataset={dataset} config={config} split={split} is valid, cache updated")
            return True
        except (DatasetNotFoundError, ConfigNotFoundError, SplitNotFoundError):
            logger.debug(
                f"the dataset={dataset}, config {config} or split {split} could not be found, don't update the cache"
            )
            return False
        except WorkerCustomError as err:
            upsert_first_rows_response(
                dataset_name=dataset,
                config_name=config,
                split_name=split,
                response=dict(err.as_response()),
                http_status=err.status_code,
                error_code=err.code,
                details=dict(err.as_response_with_cause()),
            )
            logger.debug(
                f"first-rows response for dataset={dataset} config={config} split={split} had an error, cache updated"
            )
            return False
        except Exception as err:
            e = UnexpectedError(str(err), err)
            upsert_first_rows_response(
                dataset_name=dataset,
                config_name=config,
                split_name=split,
                response=dict(e.as_response()),
                http_status=e.status_code,
                error_code=e.code,
                details=dict(e.as_response_with_cause()),
            )
            logger.debug(
                f"first-rows response for dataset={dataset} config={config} split={split} had a server"
                " error, cache updated"
            )
            return False
