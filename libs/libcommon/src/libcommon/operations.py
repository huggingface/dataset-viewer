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
    priority: Priority = Priority.NORMAL,
) -> None:
    """
    Update a dataset

    Args:
        dataset (str): the dataset
        revision (str): The revision of the dataset.
        processing_graph (ProcessingGraph): the processing graph
        priority (Priority, optional): The priority of the job. Defaults to Priority.NORMAL.

    Returns: None.
    """
    logging.debug(f"backfill {dataset=} {revision=} {priority=}")
    DatasetOrchestrator(dataset=dataset, processing_graph=processing_graph).set_revision(
        revision=revision, priority=priority, error_codes_to_retry=[]
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
