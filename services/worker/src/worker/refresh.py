# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libcache.simple_cache import (
    delete_first_rows_responses,
    get_dataset_first_rows_response_splits,
    upsert_first_rows_response,
    upsert_splits_response,
)
from libqueue.queue import add_job

from worker.responses.first_rows import get_first_rows_response
from worker.responses.splits import get_splits_response
from worker.utils import (
    ConfigNotFoundError,
    DatasetNotFoundError,
    JobType,
    SplitNotFoundError,
    UnexpectedError,
    WorkerCustomError,
)

logger = logging.getLogger(__name__)


def refresh_splits(dataset: str, hf_endpoint: str, hf_token: Optional[str] = None) -> HTTPStatus:
    try:
        response = get_splits_response(dataset, hf_endpoint, hf_token)
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
            add_job(type=JobType.FIRST_ROWS.value, dataset=d, config=c, split=s)
        logger.debug(f"{len(new_splits)} 'first-rows' jobs added for the splits of dataset={dataset}")
        return HTTPStatus.OK
    except DatasetNotFoundError as err:
        logger.debug(f"the dataset={dataset} could not be found, don't update the cache")
        return err.status_code
    except WorkerCustomError as err:
        upsert_splits_response(
            dataset,
            dict(err.as_response()),
            err.status_code,
            err.code,
            dict(err.as_response_with_cause()),
        )
        logger.debug(f"splits response for dataset={dataset} had an error, cache updated")
        return err.status_code
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
        return e.status_code


def refresh_first_rows(
    dataset: str,
    config: str,
    split: str,
    assets_base_url: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
    rows_max_bytes: Optional[int] = None,
    rows_max_number: Optional[int] = None,
    rows_min_number: Optional[int] = None,
) -> HTTPStatus:
    try:
        response = get_first_rows_response(
            dataset,
            config,
            split,
            assets_base_url=assets_base_url,
            hf_endpoint=hf_endpoint,
            hf_token=hf_token,
            max_size_fallback=max_size_fallback,
            rows_max_bytes=rows_max_bytes,
            rows_max_number=rows_max_number,
            rows_min_number=rows_min_number,
        )
        upsert_first_rows_response(dataset, config, split, dict(response), HTTPStatus.OK)
        logger.debug(f"dataset={dataset} config={config} split={split} is valid, cache updated")
        return HTTPStatus.OK
    except (DatasetNotFoundError, ConfigNotFoundError, SplitNotFoundError) as err:
        logger.debug(
            f"the dataset={dataset}, config {config} or split {split} could not be found, don't update the cache"
        )
        return err.status_code
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
        return err.status_code
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
        return e.status_code
