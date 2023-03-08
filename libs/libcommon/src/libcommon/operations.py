# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import List, Optional

from libcommon.dataset import check_support
from libcommon.exceptions import LoggedError
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Priority, Queue
from libcommon.simple_cache import DoesNotExist, delete_dataset_responses, get_response


class PreviousStepError(LoggedError):
    def __init__(self, dataset: str, step: ProcessingStep, config: Optional[str] = None, split: Optional[str] = None):
        super().__init__(
            f"Response for {step.job_type} for dataset={dataset}, config={config}, split={split} is an error."
        )


def update_dataset(
    dataset: str,
    init_processing_steps: List[ProcessingStep],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    force: bool = False,
    priority: Priority = Priority.NORMAL,
    do_check_support: bool = True,
    hf_timeout_seconds: Optional[float] = None,
) -> None:
    """
    Update a dataset

    Args:
        dataset (str): the dataset
        init_processing_steps (List[ProcessingStep]): the processing steps that must be run when updating a dataset
        hf_endpoint (str): the HF endpoint
        hf_token (Optional[str], optional): The HF token. Defaults to None.
        force (bool, optional): Force the update. Defaults to False.
        priority (Priority, optional): The priority of the job. Defaults to Priority.NORMAL.
        do_check_support (bool, optional): Check if the dataset is supported. Defaults to True.
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
    if do_check_support:
        check_support(
            dataset=dataset, hf_endpoint=hf_endpoint, hf_token=hf_token, hf_timeout_seconds=hf_timeout_seconds
        )
    logging.debug(f"refresh dataset='{dataset}'")
    queue = Queue()
    for init_processing_step in init_processing_steps:
        if init_processing_step.input_type == "dataset":
            queue.upsert_job(job_type=init_processing_step.job_type, dataset=dataset, force=force, priority=priority)


def delete_dataset(dataset: str) -> None:
    """
    Delete a dataset

    Args:
        dataset (str): the dataset

    Returns: None.
    """
    logging.debug(f"delete cache for dataset='{dataset}'")
    delete_dataset_responses(dataset=dataset)


def move_dataset(
    from_dataset: str,
    to_dataset: str,
    init_processing_steps: List[ProcessingStep],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    force: bool = False,
    priority: Priority = Priority.NORMAL,
    hf_timeout_seconds: Optional[float] = None,
) -> None:
    """
    Move a dataset

    Note that the implementation is simply to add or update the new dataset, then delete the old one in case of
    success.

    Args:
        from_dataset (str): the dataset to move
        to_dataset (str): the destination dataset
        init_processing_steps (List[ProcessingStep]): the processing steps that must be run when updating a dataset
        hf_endpoint (str): the HF endpoint
        hf_token (Optional[str], optional): The HF token. Defaults to None.
        force (bool, optional): Force the update. Defaults to False.
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
    logging.debug(f"move dataset '{from_dataset}' to '{to_dataset}'")
    update_dataset(
        dataset=to_dataset,
        init_processing_steps=init_processing_steps,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        force=force,
        priority=priority,
        hf_timeout_seconds=hf_timeout_seconds,
    )
    # ^ can raise
    delete_dataset(dataset=from_dataset)


def check_in_process(
    processing_step: ProcessingStep,
    init_processing_steps: List[ProcessingStep],
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    config: Optional[str] = None,
    split: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
) -> None:
    """Checks if the processing step is running

    Args:
        processing_step (ProcessingStep): the processing step
        init_processing_steps (List[ProcessingStep]): the processing steps that must be run when updating a dataset
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
    all_steps = processing_step.get_ancestors() + [processing_step]
    queue = Queue()
    if any(
        queue.is_job_in_process(job_type=step.job_type, dataset=dataset, config=config, split=split)
        for step in all_steps
    ):
        # the processing step, or a previous one, is still being computed
        return
    for step in processing_step.get_ancestors():
        try:
            result = get_response(kind=step.cache_kind, dataset=dataset, config=config, split=split)
        except DoesNotExist:
            # a previous step has not been computed, update the dataset
            update_dataset(
                dataset=dataset,
                init_processing_steps=init_processing_steps,
                hf_endpoint=hf_endpoint,
                hf_token=hf_token,
                force=False,
                priority=Priority.NORMAL,
                hf_timeout_seconds=hf_timeout_seconds,
            )
            return
        if result["http_status"] != HTTPStatus.OK:
            raise PreviousStepError(dataset=dataset, config=config, split=split, step=step)
    # all the dependencies (if any) have been computed successfully, the processing step should be in process
    # if the dataset is supported. Check if it is supported and update it if so.
    update_dataset(
        dataset=dataset,
        init_processing_steps=init_processing_steps,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
        force=False,
        priority=Priority.NORMAL,
        hf_timeout_seconds=hf_timeout_seconds,
    )
    return
