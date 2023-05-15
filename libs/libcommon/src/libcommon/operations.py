# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dataset import get_dataset_git_revision
from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import delete_dataset_responses
from libcommon.state import DatasetState
from libcommon.utils import Priority


def update_dataset(
    dataset: str,
    processing_graph: ProcessingGraph,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    priority: Priority = Priority.NORMAL,
    hf_timeout_seconds: Optional[float] = None,
) -> None:
    """
    Update a dataset

    Args:
        dataset (str): the dataset
        processing_graph (ProcessingGraph): the processing graph
        hf_endpoint (str): the HF endpoint
        hf_token (Optional[str], optional): The HF token. Defaults to None.
        priority (Priority, optional): The priority of the job. Defaults to Priority.NORMAL.
        hf_timeout_seconds (Optional[float], optional): The timeout for requests to the hub. None means no timeout.
          Defaults to None.

    Returns: None.

    Raises:
        - [`~exceptions.AskAccessHubRequestError`]: if the request to the Hub to get access to the
            dataset failed or timed out.
        - [`~exceptions.DatasetInfoHubRequestError`]: if the request to the Hub to get the dataset
            info failed or timed out.
        - [`~exceptions.DatasetError`]: if the dataset could not be accessed or is not supported
        - [`~exceptions.DatasetRevisionEmptyError`]
          if the current git revision (branch, commit) could not be obtained.
    """
    revision = get_dataset_git_revision(
        dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, hf_timeout_seconds=hf_timeout_seconds
    )
    logging.debug(f"refresh dataset='{dataset}'")
    backfill_dataset(dataset=dataset, processing_graph=processing_graph, revision=revision, priority=priority)


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
