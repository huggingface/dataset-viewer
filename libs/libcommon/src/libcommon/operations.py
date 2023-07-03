# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging

from libcommon.orchestrator import DatasetOrchestrator
from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import delete_dataset_responses
from libcommon.utils import Priority


def backfill_dataset(
    dataset: str,
    revision: str,
    processing_graph: ProcessingGraph,
    cache_max_days: int,
    priority: Priority = Priority.LOW,
) -> None:
    """
    Update a dataset

    Args:
        dataset (str): the dataset
        revision (str): The revision of the dataset.
        processing_graph (ProcessingGraph): the processing graph
        cache_max_days (int): the number of days to keep the cache
        priority (Priority, optional): The priority of the job. Defaults to Priority.LOW.

    Returns: None.
    """
    logging.debug(f"backfill {dataset=} {revision=} {priority=}")
    DatasetOrchestrator(dataset=dataset, processing_graph=processing_graph).set_revision(
        revision=revision, priority=priority, error_codes_to_retry=[], cache_max_days=cache_max_days
    )


def delete_dataset(dataset: str) -> None:
    """
    Delete a dataset

    Args:
        dataset (str): the dataset

    Returns: None.
    """
    logging.debug(f"delete cache for dataset='{dataset}'")
    delete_dataset_responses(dataset=dataset)
