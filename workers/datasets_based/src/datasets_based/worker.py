# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from datasets_based.config import AppConfig
from datasets_based.workers import DatasetsBasedWorker, worker_class_by_endpoint


def get_worker(app_config: AppConfig) -> DatasetsBasedWorker:
    """Get the worker for the current environment."""

    endpoint = app_config.datasets_based.endpoint
    try:
        worker = worker_class_by_endpoint[endpoint](app_config=app_config)
    except KeyError as e:
        raise ValueError(
            f"Unknown worker name '{endpoint}'. Available workers are: {list(worker_class_by_endpoint.keys())}"
        ) from e
    return worker
