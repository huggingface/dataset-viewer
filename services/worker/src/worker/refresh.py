import logging
from typing import Dict, List, Optional

from libcache.cache import (
    upsert_dataset,
    upsert_dataset_error,
    upsert_split,
    upsert_split_error,
)
from libcache.simple_cache import (
    HTTPStatus,
    delete_first_rows_responses,
    get_dataset_first_rows_response_splits,
    upsert_first_rows_response,
    upsert_splits_response,
)
from libqueue.queue import add_first_rows_job, add_split_job
from libutils.exceptions import Status400Error, Status500Error, StatusError

from worker.models.dataset import get_dataset_split_full_names
from worker.models.first_rows import get_first_rows
from worker.models.info import DatasetInfo, get_info
from worker.models.split import get_split

logger = logging.getLogger(__name__)


def refresh_dataset(dataset_name: str, hf_token: Optional[str] = None) -> None:
    try:
        split_full_names = get_dataset_split_full_names(dataset_name, hf_token)
        upsert_dataset(dataset_name, split_full_names)
        logger.debug(f"dataset={dataset_name} is valid, cache updated")
        for split_full_name in split_full_names:
            add_split_job(
                split_full_name["dataset_name"], split_full_name["config_name"], split_full_name["split_name"]
            )
    except StatusError as err:
        upsert_dataset_error(dataset_name, err)
        logger.debug(f"dataset={dataset_name} had error, cache updated")
        raise
    except Exception as err:
        upsert_dataset_error(dataset_name, Status500Error(str(err)))
        logger.debug(f"dataset={dataset_name} had error, cache updated")
        raise


def refresh_split(
    dataset_name: str,
    config_name: str,
    split_name: str,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
    rows_max_bytes: Optional[int] = None,
    rows_max_number: Optional[int] = None,
    rows_min_number: Optional[int] = None,
):
    try:
        split = get_split(
            dataset_name,
            config_name,
            split_name,
            hf_token=hf_token,
            max_size_fallback=max_size_fallback,
            rows_max_bytes=rows_max_bytes,
            rows_max_number=rows_max_number,
            rows_min_number=rows_min_number,
        )
        upsert_split(dataset_name, config_name, split_name, split)
        logger.debug(f"dataset={dataset_name} config={config_name} split={split_name} is valid, cache updated")
    except StatusError as err:
        upsert_split_error(dataset_name, config_name, split_name, err)
        logger.debug(f"dataset={dataset_name} config={config_name} split={split_name} had error, cache updated")
        raise
    except Exception as err:
        upsert_split_error(dataset_name, config_name, split_name, Status500Error(str(err)))
        logger.debug(f"dataset={dataset_name} config={config_name} split={split_name} had error, cache updated")
        raise


def get_error_response(error: StatusError) -> Dict:
    return {
        "status_code": error.status_code,
        "exception": error.exception,
        "message": error.message,
    }


def get_error_response_with_cause(error: StatusError) -> Dict:
    error_response = {
        "status_code": error.status_code,
        "exception": error.exception,
        "message": error.message,
    }
    if error.cause_exception and error.cause_message:
        error_response["cause_exception"] = error.cause_exception
        error_response["cause_message"] = error.cause_message
    if error.cause_traceback:
        error_response["cause_traceback"] = error.cause_traceback
    return error_response


def refresh_splits(dataset_name: str, hf_token: Optional[str] = None) -> HTTPStatus:
    try:
        split_full_names = get_dataset_split_full_names(dataset_name, hf_token)
        # get the number of bytes and examples for each split
        config_info: Dict[str, DatasetInfo] = {}
        splits: List[Dict] = []
        for split_full_name in split_full_names:
            try:
                dataset = split_full_name["dataset_name"]
                config = split_full_name["config_name"]
                split = split_full_name["split_name"]
                if config not in config_info:
                    config_info[config] = get_info(
                        dataset_name=split_full_name["dataset_name"],
                        config_name=split_full_name["config_name"],
                        hf_token=hf_token,
                    )
                info = config_info[config]
                num_bytes = info.splits[split].num_bytes if info.splits else None
                num_examples = info.splits[split].num_examples if info.splits else None
            except Exception:
                num_bytes = None
                num_examples = None
            splits.append(
                {
                    "dataset_name": dataset,
                    "config_name": config,
                    "split_name": split,
                    "num_bytes": num_bytes,
                    "num_examples": num_examples,
                }
            )
        response = {"splits": splits}
        upsert_splits_response(dataset_name, response, HTTPStatus.OK)
        logger.debug(f"dataset={dataset_name} is valid, cache updated")

        splits_in_cache = get_dataset_first_rows_response_splits(dataset_name)
        new_splits = [(s["dataset_name"], s["config_name"], s["split_name"]) for s in split_full_names]
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
        return HTTPStatus.OK
    except Status400Error as err:
        upsert_splits_response(dataset_name, get_error_response_with_cause(err), HTTPStatus.BAD_REQUEST)
        logger.debug(f"splits response for dataset={dataset_name} had BAD_REQUEST error, cache updated")
        return HTTPStatus.BAD_REQUEST
    except Exception as err:
        err = err if isinstance(err, Status500Error) else Status500Error(str(err))
        upsert_splits_response(
            dataset_name,
            get_error_response(err),
            HTTPStatus.INTERNAL_SERVER_ERROR,
            get_error_response_with_cause(err),
        )
        logger.debug(f"splits response for dataset={dataset_name} had INTERNAL_SERVER_ERROR error, cache updated")
        return HTTPStatus.INTERNAL_SERVER_ERROR


def refresh_first_rows(
    dataset_name: str,
    config_name: str,
    split_name: str,
    hf_token: Optional[str] = None,
    max_size_fallback: Optional[int] = None,
    rows_max_bytes: Optional[int] = None,
    rows_max_number: Optional[int] = None,
    rows_min_number: Optional[int] = None,
) -> HTTPStatus:
    try:
        response = get_first_rows(
            dataset_name,
            config_name,
            split_name,
            hf_token=hf_token,
            max_size_fallback=max_size_fallback,
            rows_max_bytes=rows_max_bytes,
            rows_max_number=rows_max_number,
            rows_min_number=rows_min_number,
        )
        upsert_first_rows_response(dataset_name, config_name, split_name, response, HTTPStatus.OK)
        logger.debug(f"dataset={dataset_name} config={config_name} split={split_name} is valid, cache updated")
        return HTTPStatus.OK
    except Status400Error as err:
        upsert_first_rows_response(
            dataset_name, config_name, split_name, get_error_response_with_cause(err), HTTPStatus.BAD_REQUEST
        )
        logger.debug(
            f"first-rows response for dataset={dataset_name} config={config_name} split={split_name} had BAD_REQUEST"
            " error, cache updated"
        )
        return HTTPStatus.BAD_REQUEST
    except Exception as err:
        err = err if isinstance(err, Status500Error) else Status500Error(str(err))
        upsert_first_rows_response(
            dataset_name,
            config_name,
            split_name,
            get_error_response(err),
            HTTPStatus.INTERNAL_SERVER_ERROR,
            get_error_response_with_cause(err),
        )
        logger.debug(
            f"first-rows response for dataset={dataset_name} config={config_name} split={split_name} had"
            " INTERNAL_SERVER_ERROR error, cache updated"
        )
        return HTTPStatus.INTERNAL_SERVER_ERROR
