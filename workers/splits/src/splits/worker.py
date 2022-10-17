# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libcache.simple_cache import (
    delete_first_rows_responses,
    get_dataset_first_rows_response_splits,
    upsert_splits_response,
)
from libqueue.worker import Worker

from splits.response import get_splits_response
from splits.utils import (
    DatasetNotFoundError,
    Queues,
    UnexpectedError,
    WorkerCustomError,
)

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
            sleep_seconds=sleep_seconds,
            max_memory_pct=max_memory_pct,
            max_load_pct=max_load_pct,
        )
        self._queues = Queues(max_jobs_per_dataset=max_jobs_per_dataset)
        self.hf_endpoint = hf_endpoint
        self.hf_token = hf_token

    @property
    def queue(self):
        return self._queues.splits

    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
    ) -> bool:
        try:
            response = get_splits_response(dataset, self.hf_endpoint, self.hf_token)
            upsert_splits_response(dataset, dict(response), HTTPStatus.OK)
            logger.debug(f"dataset={dataset} is valid, cache updated")

            splits_in_cache = get_dataset_first_rows_response_splits(dataset)
            new_splits = [(s["dataset"], s["config"], s["split"]) for s in response["splits"]]
            splits_to_delete = [s for s in splits_in_cache if s not in new_splits]
            for d, c, s in splits_to_delete:
                delete_first_rows_responses(d, c, s)
            logger.debug(
                f"{len(splits_to_delete)} 'first-rows' responses deleted from the cache for obsolete splits of"
                f" dataset={dataset}"
            )
            for d, c, s in new_splits:
                self._queues.first_rows.add_job(dataset=d, config=c, split=s)
            logger.debug(f"{len(new_splits)} 'first-rows' jobs added for the splits of dataset={dataset}")
            return True
        except DatasetNotFoundError:
            logger.debug(f"the dataset={dataset} could not be found, don't update the cache")
            return False
        except WorkerCustomError as err:
            upsert_splits_response(
                dataset,
                dict(err.as_response()),
                err.status_code,
                err.code,
                dict(err.as_response_with_cause()),
            )
            logger.debug(f"splits response for dataset={dataset} had an error, cache updated")
            return False
        except Exception as err:
            e = UnexpectedError(str(err), err)
            upsert_splits_response(
                dataset,
                dict(e.as_response()),
                e.status_code,
                e.code,
                dict(e.as_response_with_cause()),
            )
            logger.debug(f"splits response for dataset={dataset} had a server error, cache updated")
            return False
