# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dataset import get_dataset_git_revision
from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import delete_dataset_responses
from libcommon.state import DatasetState
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
    dataset_state = DatasetState(
        dataset=dataset, revision=revision, processing_graph=processing_graph, priority=priority
    )
    dataset_state.backfill()


def delete_dataset(dataset: str) -> None:
    """
    Delete a dataset

    Args:
        dataset (str): the dataset

    Returns: None.
    """
    logging.debug(f"delete cache for dataset='{dataset}'")
    delete_dataset_responses(dataset=dataset)
