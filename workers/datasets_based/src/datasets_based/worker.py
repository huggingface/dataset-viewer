# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Mapping, Type, Union

from datasets_based.config import AppConfig
from datasets_based.workers.first_rows import FirstRowsWorker
from datasets_based.workers.parquet import ParquetWorker
from datasets_based.workers.splits import SplitsWorker

DatasetsBasedWorker = Union[SplitsWorker, FirstRowsWorker, ParquetWorker]


def get_worker(app_config: AppConfig) -> DatasetsBasedWorker:
    """Get the worker for the current environment."""

    datasets_based_worker_classes: Mapping[str, Type[DatasetsBasedWorker]] = {
        "/splits": SplitsWorker,
        "/first-rows": FirstRowsWorker,
        "/parquet": ParquetWorker,
    }
    try:
        endpoint = app_config.datasets_based.endpoint
        worker = datasets_based_worker_classes[endpoint](app_config=app_config, endpoint=endpoint)
    except KeyError as e:
        raise ValueError(
            f"Unknown worker name '{endpoint}'. Available workers are: {list(datasets_based_worker_classes.keys())}"
        ) from e
    return worker
