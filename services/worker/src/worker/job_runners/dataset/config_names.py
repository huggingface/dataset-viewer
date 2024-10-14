# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from datasets import (
    get_dataset_config_names,
    get_dataset_default_config_name,
)
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from datasets.exceptions import (
    DataFilesNotFoundError as _DataFilesNotFoundError,
)
from datasets.exceptions import DatasetNotFoundError
from huggingface_hub.utils import HfHubHTTPError
from libcommon.exceptions import (
    ConfigNamesError,
    DataFilesNotFoundError,
    DatasetWithScriptNotSupportedError,
    DatasetWithTooManyConfigsError,
    EmptyDatasetError,
    FileFormatMismatchBetweenSplitsError,
    RetryableConfigNamesError,
)

from worker.dtos import CompleteJobResult, ConfigNameItem, DatasetConfigNamesResponse
from worker.job_runners.dataset.dataset_job_runner import (
    DatasetJobRunnerWithDatasetsCache,
)


def compute_config_names_response(
    dataset: str,
    max_number: int,
    hf_token: Optional[str] = None,
) -> DatasetConfigNamesResponse:
    """
    Get the response of 'dataset-config-names' for one specific dataset on huggingface.co.
    Dataset can be gated if you pass an acceptable token.
    It is assumed that the dataset exists and can be accessed using the token.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        max_number (`int`):
            The maximum number of configs for a dataset.
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)

    Raises:
        [~`libcommon.exceptions.EmptyDatasetError`]:
          The dataset is empty.
        [~`libcommon.exceptions.ConfigNamesError`]:
          If the list of configs could not be obtained using the datasets library.
        [~`libcommon.exceptions.DatasetWithScriptNotSupportedError`]:
            If the dataset has a dataset script.

    Returns:
        `DatasetConfigNamesResponse`: An object with the list of config names.
    """
    logging.info(f"compute 'dataset-config-names' for {dataset=}")
    # get the list of splits in streaming mode
    try:
        default_config_name: Optional[str] = None
        config_names = get_dataset_config_names(
            path=dataset,
            token=hf_token,
        )
        if len(config_names) > 1:
            default_config_name = get_dataset_default_config_name(
                path=dataset,
                token=hf_token,
            )
        config_name_items: list[ConfigNameItem] = [
            {"dataset": dataset, "config": str(config)}
            for config in sorted(
                config_names,
                key=lambda config_name: (config_name != default_config_name, config_name),  # default config first
            )
        ]
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except _DataFilesNotFoundError as err:
        raise DataFilesNotFoundError(str(err), cause=err) from err
    except ValueError as err:
        if "trust_remote_code" in str(err):
            raise DatasetWithScriptNotSupportedError from err
        if "Couldn't infer the same data file format for all splits" in str(err):
            raise FileFormatMismatchBetweenSplitsError(str(err), cause=err) from err
        raise ConfigNamesError("Cannot get the config names for the dataset.", cause=err) from err
    except (HfHubHTTPError, BrokenPipeError, DatasetNotFoundError, PermissionError, ConnectionError) as err:
        raise RetryableConfigNamesError("Cannot get the config names for the dataset.", cause=err) from err
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

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_config_names_response(
                dataset=self.dataset,
                hf_token=self.app_config.common.hf_token,
                max_number=self.app_config.config_names.max_number,
            )
        )
