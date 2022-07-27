import logging
from http import HTTPStatus
from typing import Optional, Tuple

from libcache.simple_cache import (
    delete_first_rows_responses,
    get_dataset_first_rows_response_splits,
    upsert_first_rows_response,
    upsert_splits_response,
)
from libqueue.queue import add_first_rows_job

from worker.responses.first_rows import get_first_rows_response
from worker.responses.splits import get_splits_response
from worker.utils import (
    ConfigNotFoundError,
    DatasetNotFoundError,
    SplitNotFoundError,
    UnexpectedError,
    WorkerCustomError,
)

logger = logging.getLogger(__name__)


def refresh_splits(dataset_name: str, hf_token: Optional[str] = None) -> Tuple[HTTPStatus, bool]:
    try:
        response = get_splits_response(dataset_name, hf_token)
        upsert_splits_response(dataset_name, dict(response), HTTPStatus.OK)
        logger.debug(f"dataset={dataset_name} is valid, cache updated")

        splits_in_cache = get_dataset_first_rows_response_splits(dataset_name)
        new_splits = [(s["dataset_name"], s["config_name"], s["split_name"]) for s in response["splits"]]
        splits_to_delete = [s for s in splits_in_cache if s not in new_splits]
        for d, c, s in splits_to_delete:
            delete_first_rows_responses(d, c, s)
        logger.debug(
            f"{len(splits_to_delete)} 'first-rows' responses deleted from the cache for obsolete splits of"
            f" dataset={dataset_name}"
        )
        for d, c, s in new_splits:
            add_first_rows_job(d, c, s)
        logger.debug(f"{len(new_splits)} 'first-rows' jobs added for the splits of dataset={dataset_name}")
        return HTTPStatus.OK, False
    except WorkerCustomError as err:
        upsert_splits_response(
            dataset_name,
            dict(err.as_response()),
            err.status_code,
            err.code,
            dict(err.as_response_with_cause()),
        )
        logger.debug(f"splits response for dataset={dataset_name} had an error, cache updated")
        return err.status_code, False
    except Exception as err:
        e = UnexpectedError(str(err), err)
        upsert_splits_response(
            dataset_name,
            dict(e.as_response()),
            e.status_code,
            e.code,
            dict(e.as_response_with_cause()),
        )
        logger.debug(f"splits response for dataset={dataset_name} had a server error, cache updated")
        return e.status_code, True


def refresh_first_rows(
    dataset_name: str,
    config_name: str,
    split_name: str,
    assets_base_url: str,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
    rows_max_bytes: Optional[int] = None,
    rows_max_number: Optional[int] = None,
    rows_min_number: Optional[int] = None,
) -> Tuple[HTTPStatus, bool]:
    try:
        response = get_first_rows_response(
            dataset_name,
            config_name,
            split_name,
            assets_base_url=assets_base_url,
            hf_token=hf_token,
            max_size_fallback=max_size_fallback,
            rows_max_bytes=rows_max_bytes,
            rows_max_number=rows_max_number,
            rows_min_number=rows_min_number,
        )
        upsert_first_rows_response(dataset_name, config_name, split_name, dict(response), HTTPStatus.OK)
        logger.debug(f"dataset={dataset_name} config={config_name} split={split_name} is valid, cache updated")
        return HTTPStatus.OK, False
    except WorkerCustomError as err:
        upsert_first_rows_response(
            dataset_name,
            config_name,
            split_name,
            dict(err.as_response()),
            err.status_code,
            err.code,
            dict(err.as_response_with_cause()),
        )
        logger.debug(
            f"first-rows response for dataset={dataset_name} config={config_name} split={split_name} had an error,"
            " cache updated"
        )
        return err.status_code, False
    except Exception as err:
        e = UnexpectedError(str(err), err)
        upsert_first_rows_response(
            dataset_name,
            config_name,
            split_name,
            dict(e.as_response()),
            e.status_code,
            e.code,
            dict(e.as_response_with_cause()),
        )
        logger.debug(
            f"first-rows response for dataset={dataset_name} config={config_name} split={split_name} had a server"
            " error, cache updated"
        )
        return e.status_code, True
