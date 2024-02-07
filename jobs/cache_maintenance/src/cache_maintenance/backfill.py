# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dtos import Priority
from libcommon.exceptions import NotSupportedError
from libcommon.operations import backfill_dataset
from libcommon.simple_cache import get_all_datasets
from libcommon.storage_client import StorageClient


def backfill_cache(
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> None:
    logging.info("backfill datasets in the database and delete non-supported ones")
    datasets_in_database = get_all_datasets()
    logging.info(f"analyzing {len(datasets_in_database)} datasets in the database")
    analyzed_datasets = 0
    supported_datasets = 0
    deleted_datasets = 0
    error_datasets = 0
    log_batch = 100

    def get_log() -> str:
        return (
            f"{analyzed_datasets} analyzed datasets (total: {len(datasets_in_database)} datasets):"
            f" {deleted_datasets} datasets have been deleted ({100 * deleted_datasets / analyzed_datasets:.2f}%),"
            f" {error_datasets} datasets raised an exception ({100 * error_datasets / analyzed_datasets:.2f}%)"
        )

    for dataset in datasets_in_database:
        analyzed_datasets += 1
        try:
            backfill_dataset(
                dataset=dataset,
                hf_endpoint=hf_endpoint,
                blocked_datasets=blocked_datasets,
                hf_token=hf_token,
                priority=Priority.LOW,
                hf_timeout_seconds=None,
                storage_clients=storage_clients,
            )
            supported_datasets += 1
        except NotSupportedError:
            deleted_datasets += 1
        except Exception as e:
            logging.warning(f"failed to update_dataset {dataset}: {e}")
            error_datasets += 1
        logging.debug(get_log())
        if analyzed_datasets % log_batch == 0:
            logging.info(get_log())

    logging.info(get_log())
    logging.info("backfill completed")
