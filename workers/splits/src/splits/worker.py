# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import importlib.metadata
import logging
from http import HTTPStatus
from typing import Any, Dict, List, Literal, Mapping, Optional, TypedDict, Union

from datasets import (
    DatasetInfo,
    get_dataset_config_info,
    get_dataset_config_names,
    get_dataset_split_names,
)
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from libcommon.exceptions import CustomError
from libcommon.processing_steps import first_rows_step, splits_step
from libcommon.simple_cache import delete_response, get_dataset_response_ids
from libcommon.worker import Queue, Worker

from splits.config import WorkerConfig

SplitsWorkerErrorCode = Literal[
    "EmptyDatasetError",
    "SplitsNamesError",
]


class SplitWorkerError(CustomError):
    """Base class for worker exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: SplitsWorkerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=str(code), cause=cause, disclose_cause=disclose_cause
        )


class SplitsNamesError(SplitWorkerError):
    """Raised when the split names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "SplitsNamesError", cause, True)


class EmptyDatasetError(SplitWorkerError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class SplitFullName(TypedDict):
    dataset: str
    config: str
    split: str


class SplitItem(SplitFullName):
    num_bytes: Optional[int]
    num_examples: Optional[int]


class SplitsResponseContent(TypedDict):
    splits: List[SplitItem]


def get_dataset_split_full_names(dataset: str, use_auth_token: Union[bool, str, None] = False) -> List[SplitFullName]:
    """Get the list of splits full names (split and config) for a dataset.

    Args:
        dataset (str): A dataset name. If the repository is namespaced (a user or an organization), the namespace and
          the dataset name are separated with a slash (`/`), for example: `user/dataset`.
        use_auth_token (Union[bool, str, None], optional): user token. It allows to retrieve the splits for gated
          datasets. Defaults to False (no authentication).

    Returns:
        List[SplitFullName]: a list of splits full names: objects with the keys `dataset`, `config` and `split`. They
          are sorted alphabetically by configuration (config), but the splits order for a given configuration is
          preserved.
    """
    logging.info(f"get dataset '{dataset}' split full names")
    return [
        {"dataset": dataset, "config": config, "split": split}
        for config in sorted(get_dataset_config_names(path=dataset, use_auth_token=use_auth_token))
        for split in get_dataset_split_names(path=dataset, config_name=config, use_auth_token=use_auth_token)
    ]


def compute_splits_response(
    dataset: str,
    hf_token: Optional[str] = None,
) -> SplitsResponseContent:
    """
    Get the response of /splits for one specific dataset on huggingface.co.
    Dataset can be private or gated if you pass an acceptable token.

    It is assumed that the dataset exist and can be accessed using the token.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
    Returns:
        `SplitsResponseResult`: An object with the splits_response
          (list of splits names) and the dataset_git_revision (sha) if any.
    <Tip>
    Raises the following errors:
        - [`~worker.exceptions.DatasetNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
        - [`~worker.exceptions.SplitsNamesError`]
          If the list of splits could not be obtained using the datasets library.
    </Tip>
    """
    logging.info(f"get splits for dataset={dataset}")
    use_auth_token: Union[bool, str, None] = hf_token if hf_token is not None else False
    # get the list of splits
    try:
        split_full_names = get_dataset_split_full_names(dataset=dataset, use_auth_token=use_auth_token)
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except Exception as err:
        raise SplitsNamesError("Cannot get the split names for the dataset.", cause=err) from err
    # get the number of bytes and examples for each split
    config_info: Dict[str, DatasetInfo] = {}
    split_items: List[SplitItem] = []
    for split_full_name in split_full_names:
        dataset = split_full_name["dataset"]
        config = split_full_name["config"]
        split = split_full_name["split"]
        try:
            if config not in config_info:
                config_info[config] = get_dataset_config_info(
                    path=dataset,
                    config_name=config,
                    use_auth_token=use_auth_token,
                )
            info = config_info[config]
            num_bytes = info.splits[split].num_bytes if info.splits else None
            num_examples = info.splits[split].num_examples if info.splits else None
        except Exception:
            num_bytes = None
            num_examples = None
        split_items.append(
            {
                "dataset": dataset,
                "config": config,
                "split": split,
                "num_bytes": num_bytes,
                "num_examples": num_examples,
            }
        )
    return {"splits": split_items}


class SplitsWorker(Worker):
    def __init__(self, worker_config: WorkerConfig):
        super().__init__(
            processing_step=splits_step,
            common_config=worker_config.common,
            queue_config=worker_config.queue,
            version=importlib.metadata.version(__package__),
        )

    def compute(
        self,
        dataset: str,
        config: Optional[str] = None,
        split: Optional[str] = None,
        force: bool = False,
    ) -> Mapping[str, Any]:
        content = compute_splits_response(dataset=dataset, hf_token=self.common_config.hf_token)

        # TODO: create a "hook" instead, so that we don't refer to first_rows here
        new_splits = [(s["dataset"], s["config"], s["split"]) for s in content["splits"]]
        # remove obsolete first-rows responses from the cache
        first_rows_responses_in_cache = [
            (s["dataset"], s["config"], s["split"])
            for s in get_dataset_response_ids(dataset=dataset)
            if s["kind"] == first_rows_step.cache_kind
        ]
        first_rows_responses_to_delete = [s for s in first_rows_responses_in_cache if s not in new_splits]
        for d, c, s in first_rows_responses_to_delete:
            delete_response(kind=first_rows_step.cache_kind, dataset=d, config=c, split=s)
        logging.debug(
            f"{len(first_rows_responses_to_delete)} 'first-rows' responses deleted from the cache for obsolete"
            f" splits of dataset={dataset}"
        )
        # compute the 'first-rows' responses for the new splits
        for d, c, s in new_splits:
            # we force the refresh of the /first_rows responses if the /splits refresh was forced
            Queue(type=first_rows_step.job_type).add_job(dataset=d, config=c, split=s, force=force)
        logging.debug(f"{len(new_splits)} 'first-rows' jobs added for the splits of dataset={dataset}")

        return content
