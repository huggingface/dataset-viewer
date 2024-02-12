# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from dataclasses import dataclass
from typing import Optional

from libcommon.dtos import Priority
from libcommon.operations import OperationsStatistics, backfill_dataset
from libcommon.simple_cache import get_all_datasets
from libcommon.storage_client import StorageClient


@dataclass
class BackfillStatistics:
    num_total_datasets: int = 0
    num_analyzed_datasets: int = 0
    num_error_datasets: int = 0
    operations: OperationsStatistics = OperationsStatistics()

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


def backfill_cache(
    hf_endpoint: str,
    blocked_datasets: list[str],
    hf_token: Optional[str] = None,
    storage_clients: Optional[list[StorageClient]] = None,
) -> None:
    logging.info("backfill datasets in the database and delete non-supported ones")
    datasets_in_database = get_all_datasets()
    logging.info(f"analyzing {len(datasets_in_database)} datasets in the database")
    statistics = BackfillStatistics(num_total_datasets=len(datasets_in_database))
    log_batch = 100

    for dataset in datasets_in_database:
        try:
            statistics.add(
                BackfillStatistics(
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
            )
        except Exception as e:
            logging.warning(f"failed to update_dataset {dataset}: {e}")
            statistics.add(BackfillStatistics(num_analyzed_datasets=1, num_error_datasets=1))
        logging.debug(statistics.get_log())
        if statistics.num_analyzed_datasets % log_batch == 0:
            logging.info(statistics.get_log())

    logging.info(statistics.get_log())
    logging.info("backfill completed")
