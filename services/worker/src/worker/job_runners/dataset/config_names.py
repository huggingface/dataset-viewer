# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import List, Optional, TypedDict, Union

from datasets import get_dataset_config_names
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from libcommon.constants import PROCESSING_STEP_CONFIG_NAMES_VERSION
from libcommon.exceptions import (
    ConfigNamesError,
    DatasetModuleNotInstalledError,
    EmptyDatasetError,
)

from worker.job_runners.dataset.dataset_job_runner import DatasetCachedJobRunner
from worker.utils import CompleteJobResult


class ConfigNameItem(TypedDict):
    dataset: str
    config: str


class ConfigNamesResponse(TypedDict):
    config_names: List[ConfigNameItem]


def compute_config_names_response(
    dataset: str,
    hf_token: Optional[str] = None,
) -> ConfigNamesResponse:
    """
    Get the response of /config-names for one specific dataset on huggingface.co.
    Dataset can be private or gated if you pass an acceptable token.

    It is assumed that the dataset exists and can be accessed using the token.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
    Returns:
        `ConfigNamesResponse`: An object with the list of config names.
    <Tip>
    Raises the following errors:
        - [`~job_runners.config_names.EmptyDatasetError`]
          The dataset is empty.
        - [`~job_runners.config_names.DatasetModuleNotInstalledError`]
          The dataset tries to import a module that is not installed.
        - [`~job_runners.config_names.ConfigNamesError`]
          If the list of configs could not be obtained using the datasets library.
    </Tip>
    """
    logging.info(f"get config names for dataset={dataset}")
    use_auth_token: Union[bool, str, None] = hf_token if hf_token is not None else False
    # get the list of splits in streaming mode
    try:
        config_name_items: List[ConfigNameItem] = [
            {"dataset": dataset, "config": str(config)}
            for config in sorted(get_dataset_config_names(path=dataset, use_auth_token=use_auth_token))
        ]
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except ImportError as err:
        raise DatasetModuleNotInstalledError(
            "The dataset tries to import a module that is not installed.", cause=err
        ) from err
    except Exception as err:
        raise ConfigNamesError("Cannot get the config names for the dataset.", cause=err) from err
    return ConfigNamesResponse(config_names=config_name_items)


class ConfigNamesJobRunner(DatasetCachedJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/config-names"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_NAMES_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_config_names_response(dataset=self.dataset, hf_token=self.app_config.common.hf_token)
        )
