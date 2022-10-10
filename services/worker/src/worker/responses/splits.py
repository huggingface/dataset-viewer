# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Dict, List, Optional, TypedDict, Union

from datasets import (
    DatasetInfo,
    get_dataset_config_info,
    get_dataset_config_names,
    get_dataset_split_names,
)
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from huggingface_hub.hf_api import HfApi  # type: ignore
from huggingface_hub.utils import RepositoryNotFoundError  # type: ignore

from worker.utils import DatasetNotFoundError, EmptyDatasetError, SplitsNamesError

logger = logging.getLogger(__name__)


class SplitFullName(TypedDict):
    dataset: str
    config: str
    split: str


class SplitItem(SplitFullName):
    num_bytes: Optional[int]
    num_examples: Optional[int]


class SplitsResponse(TypedDict):
    splits: List[SplitItem]


def get_dataset_split_full_names(dataset: str, use_auth_token: Union[bool, str, None] = False) -> List[SplitFullName]:
    logger.info(f"get dataset '{dataset}' split full names")
    return [
        {"dataset": dataset, "config": config, "split": split}
        for config in get_dataset_config_names(dataset, use_auth_token=use_auth_token)
        for split in get_dataset_split_names(dataset, config, use_auth_token=use_auth_token)
    ]


def get_splits_response(
    dataset: str,
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> SplitsResponse:
    """
    Get the response of /splits for one specific dataset on huggingface.co.
    Dataset can be private or gated if you pass an acceptable token.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
    Returns:
        [`SplitsResponse`]: The list of splits names.
    <Tip>
    Raises the following errors:
        - [`~worker.exceptions.DatasetNotFoundError`]
          If the repository to download from cannot be found. This may be because it doesn't exist,
          or because it is set to `private` and you do not have access.
        - [`~worker.exceptions.SplitsNamesError`]
          If the list of splits could not be obtained using the datasets library.
    </Tip>
    """
    logger.info(f"get splits for dataset={dataset}")
    use_auth_token: Union[bool, str, None] = hf_token if hf_token is not None else False
    # first try to get the dataset config info
    try:
        HfApi(endpoint=hf_endpoint).dataset_info(dataset, use_auth_token=use_auth_token)
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub.") from err
    # get the list of splits
    try:
        split_full_names = get_dataset_split_full_names(dataset, use_auth_token)
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
