# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Optional

from libcommon.dataset import get_dataset_git_revision
from libcommon.exceptions import LoggedError
from libcommon.processing_graph import ProcessingGraph
from libcommon.queue import Priority, Queue
from libcommon.simple_cache import DoesNotExist, delete_dataset_responses, get_response
from libcommon.state import DatasetState


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


def check_in_process(
    processing_step_name: str,
    processing_graph: ProcessingGraph,
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    config: Optional[str] = None,
    split: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> None:
    """Checks if the processing step is running

    Args:
        processing_step_name (str): the name of the processing step
        processing_graph (ProcessingGraph): the processing graph
        dataset (str): the dataset
        hf_endpoint (str): the HF endpoint
        hf_token (Optional[str], optional): The HF token. Defaults to None.
        config (Optional[str], optional): The config, if any. Defaults to None.
        split (Optional[str], optional): The split, if any. Defaults to None.
        hf_timeout_seconds (Optional[float], optional): The timeout for requests to the hub. None means no timeout.
          Defaults to None.


    Returns: None. Does not raise if the processing step is running.

    Raises:
        - [`~libcommon.dataset.AskAccessHubRequestError`]: if the request to the Hub to get access to the
            dataset failed or timed out.
        - [`~libcommon.dataset.DatasetInfoHubRequestError`]: if the request to the Hub to get the dataset
            info failed or timed out.
        - [`~libcommon.operations.PreviousStepError`]: a previous step has an error
        - [`~libcommon.dataset.DatasetError`]: if the dataset could not be accessed or is not supported
    """
    processing_step = processing_graph.get_processing_step(processing_step_name)
    ancestors = processing_graph.get_ancestors(processing_step_name)
    queue = Queue()
    if any(
        queue.is_job_in_process(
            job_type=ancestor_or_processing_step.job_type, dataset=dataset, config=config, split=split
        )
        for ancestor_or_processing_step in ancestors + [processing_step]
    ):
        # the processing step, or a previous one, is still being computed
        return
    for ancestor in ancestors:
        try:
            result = get_response(kind=ancestor.cache_kind, dataset=dataset, config=config, split=split)
        except DoesNotExist:
            # a previous step has not been computed, update the dataset
            update_dataset(
                dataset=dataset,
                processing_graph=processing_graph,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                priority=Priority.NORMAL,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            return
        if result["http_status"] != HTTPStatus.OK:
            raise PreviousStepError(dataset=dataset, config=config, split=split, job_type=ancestor.job_type)
    # all the dependencies (if any) have been computed successfully, the processing step should be in process
    # if the dataset is supported. Check if it is supported and update it if so.
    update_dataset(
        dataset=dataset,
        processing_graph=processing_graph,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        priority=Priority.NORMAL,
        hf_timeout_seconds=hf_timeout_seconds,
    )
    return
