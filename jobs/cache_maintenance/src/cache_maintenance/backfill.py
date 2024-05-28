# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import Optional

from libcommon.dtos import Priority
from libcommon.operations import OperationsStatistics, backfill_dataset
from libcommon.simple_cache import get_all_datasets, get_datasets_with_retryable_errors
from libcommon.storage_client import StorageClient

MAX_BACKFILL_WORKERS = 8
LOG_BATCH = 100


@dataclass
class BackfillStatistics:
    num_total_datasets: int = 0
    num_analyzed_datasets: int = 0
    num_error_datasets: int = 0
    operations: OperationsStatistics = field(default_factory=OperationsStatistics)

    def add(self, other: "BackfillStatistics") -> None:
        self.num_total_datasets += other.num_total_datasets
        self.num_analyzed_datasets += other.num_analyzed_datasets
        self.num_error_datasets += other.num_error_datasets
        self.operations.add(other.operations)

    def get_log(self) -> str:
        return (
            f"{self.num_analyzed_datasets} analyzed datasets (total: {self.num_total_datasets} datasets): "
            f"{self.operations.num_untouched_datasets} already ok ({100 * self.operations.num_untouched_datasets / self.num_analyzed_datasets:.2f}%), "
            f"{self.operations.num_backfilled_datasets} backfilled ({100 * self.operations.num_backfilled_datasets / self.num_analyzed_datasets:.2f}%), "
            f"{self.operations.num_deleted_datasets} deleted ({100 * self.operations.num_deleted_datasets / self.num_analyzed_datasets:.2f}%), "
            f"{self.num_error_datasets} raised an exception ({100 * self.num_error_datasets / self.num_analyzed_datasets:.2f}%). "
            f"{self.operations.tasks.get_log()}"
        )


def backfill_all_datasets(
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> None:
    logging.info("backfill datasets in the database and delete non-supported ones")
    datasets_in_database = get_all_datasets()
    backfill_datasets(
        dataset_names=datasets_in_database,
        hf_endpoint=hf_endpoint,
        blocked_datasets=blocked_datasets,
        hf_token=hf_token,
        storage_clients=storage_clients,
    )


def backfill_retryable_errors(
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> None:
    logging.info("backfill datasets that have a retryable error")
    dataset_names = get_datasets_with_retryable_errors()
    backfill_datasets(
        dataset_names=dataset_names,
        hf_endpoint=hf_endpoint,
        blocked_datasets=blocked_datasets,
        hf_token=hf_token,
        storage_clients=storage_clients,
    )


def try_backfill_dataset(
    dataset: str,
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> BackfillStatistics:
    try:
        return BackfillStatistics(
            num_analyzed_datasets=1,
            operations=backfill_dataset(
                dataset=dataset,
                hf_endpoint=hf_endpoint,
                blocked_datasets=blocked_datasets,
                hf_token=hf_token,
                priority=Priority.LOW,
                hf_timeout_seconds=None,
                storage_clients=storage_clients,
            ),
        )

    except Exception as e:
        logging.warning(f"failed to update_dataset {dataset}: {e}")
        return BackfillStatistics(num_analyzed_datasets=1, num_error_datasets=1)


def backfill_datasets(
    dataset_names: set[str],
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> BackfillStatistics:
    logging.info(f"analyzing {len(dataset_names)} datasets in the database")

    statistics = BackfillStatistics(num_total_datasets=len(dataset_names))

    def _backfill_dataset(dataset: str) -> BackfillStatistics:
        # all the parameters are common, but the dataset is different
        return try_backfill_dataset(dataset, hf_endpoint, blocked_datasets, hf_token, storage_clients)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_BACKFILL_WORKERS) as executor:
        def get_futures():
            for dataset in dataset_names:
                yield executor.submit(_backfill_dataset, dataset)

        # Start the load operations and gives stats on the progress
        for future in concurrent.futures.as_completed(get_futures()):
            try:
                dataset_statistics = future.result()
            except Exception as e:
                logging.warning(f"Unexpected error: {e}")
                dataset_statistics = BackfillStatistics(
                    num_total_datasets=1, num_analyzed_datasets=1, num_error_datasets=1
                )
            finally:
                statistics.add(dataset_statistics)
                logging.debug(statistics.get_log())
                if statistics.num_analyzed_datasets % LOG_BATCH == 0:
                    logging.info(statistics.get_log())

    logging.info(statistics.get_log())
    logging.info("backfill completed")

    return statistics
