# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from datasets import get_dataset_config_names
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from libcommon.constants import PROCESSING_STEP_DATASET_CONFIG_NAMES_VERSION
from libcommon.exceptions import (
    ConfigNamesError,
    DatasetModuleNotInstalledError,
    DatasetWithTooManyConfigsError,
    EmptyDatasetError,
)

from worker.dtos import CompleteJobResult, ConfigNameItem, DatasetConfigNamesResponse
from worker.job_runners.dataset.dataset_job_runner import (
    DatasetJobRunnerWithDatasetsCache,
)
from worker.utils import disable_dataset_scripts_support


def compute_config_names_response(
    dataset: str,
    max_number: int,
    dataset_scripts_allow_list: list[str],
    hf_token: Optional[str] = None,
) -> DatasetConfigNamesResponse:
    """
    Get the response of dataset-config-names for one specific dataset on huggingface.co.
    Dataset can be private or gated if you pass an acceptable token.

    It is assumed that the dataset exists and can be accessed using the token.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        dataset_scripts_allow_list (`list[str]`):
            List of datasets for which we support dataset scripts.
            Unix shell-style wildcards also work in the dataset name for namespaced datasets,
            for example `some_namespace/*` to refer to all the datasets in the `some_namespace` namespace.
            The keyword `{{ALL_DATASETS_WITH_NO_NAMESPACE}}` refers to all the datasets without namespace.
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
    Returns:
        `DatasetConfigNamesResponse`: An object with the list of config names.
    Raises the following errors:
        - [`libcommon.exceptions.EmptyDatasetError`]
          The dataset is empty.
        - [`libcommon.exceptions.DatasetModuleNotInstalledError`]
          The dataset tries to import a module that is not installed.
        - [`libcommon.exceptions.ConfigNamesError`]
          If the list of configs could not be obtained using the datasets library.
        - [`libcommon.exceptions.DatasetWithScriptNotSupportedError`]
            If the dataset has a dataset script and is not in the allow list.
    """
    logging.info(f"get config names for dataset={dataset}")
    # get the list of splits in streaming mode
    try:
        with disable_dataset_scripts_support(allow_list=dataset_scripts_allow_list):
            config_name_items: list[ConfigNameItem] = [
                {"dataset": dataset, "config": str(config)}
                for config in sorted(get_dataset_config_names(path=dataset, token=hf_token))
            ]
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except ImportError as err:
        raise DatasetModuleNotInstalledError(
            "The dataset tries to import a module that is not installed.", cause=err
        ) from err
    except Exception as err:
        raise ConfigNamesError("Cannot get the config names for the dataset.", cause=err) from err

    number_of_configs = len(config_name_items)
    if number_of_configs > max_number:
        raise DatasetWithTooManyConfigsError(
            f"The maximum number of configs allowed is {max_number}, dataset has {number_of_configs} configs."
        )

    return DatasetConfigNamesResponse(config_names=config_name_items)


class DatasetConfigNamesJobRunner(DatasetJobRunnerWithDatasetsCache):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-config-names"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_DATASET_CONFIG_NAMES_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_config_names_response(
                dataset=self.dataset,
                hf_token=self.app_config.common.hf_token,
                max_number=self.app_config.config_names.max_number,
                dataset_scripts_allow_list=self.app_config.common.dataset_scripts_allow_list,
            )
        )
