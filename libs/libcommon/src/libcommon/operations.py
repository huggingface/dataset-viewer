# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.dataset import get_dataset_git_revision
from libcommon.exceptions import LoggedError
from libcommon.processing_graph import ProcessingGraph
from libcommon.simple_cache import delete_dataset_responses
from libcommon.state import DatasetState
from libcommon.utils import Priority


class PreviousStepError(LoggedError):
    def __init__(self, dataset: str, job_type: str, config: Optional[str] = None, split: Optional[str] = None):
        super().__init__(f"Response for {job_type} for dataset={dataset}, config={config}, split={split} is an error.")


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
        - [`~libcommon.dataset.AskAccessHubRequestError`]: if the request to the Hub to get access to the
            dataset failed or timed out.
        - [`~libcommon.dataset.DatasetInfoHubRequestError`]: if the request to the Hub to get the dataset
            info failed or timed out.
        - [`~libcommon.dataset.DatasetError`]: if the dataset could not be accessed or is not supported
    """
    revision = get_dataset_git_revision(
        dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, hf_timeout_seconds=hf_timeout_seconds
    )
    logging.debug(f"refresh dataset='{dataset}'")
    backfill_dataset(dataset=dataset, processing_graph=processing_graph, revision=revision, priority=priority)


def backfill_dataset(
    dataset: str,
    processing_graph: ProcessingGraph,
    revision: Optional[str] = None,
    priority: Priority = Priority.NORMAL,
) -> None:
    """
    Update a dataset

    Args:
        dataset (str): the dataset
        processing_graph (ProcessingGraph): the processing graph
        revision (str, optional): The revision of the dataset. Defaults to None.
        priority (Priority, optional): The priority of the job. Defaults to Priority.NORMAL.

    Returns: None.
    """
    logging.debug(f"backfill {dataset=} {revision=} {priority=}")
    dataset_state = DatasetState(
        dataset=dataset, processing_graph=processing_graph, priority=priority, revision=revision
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
