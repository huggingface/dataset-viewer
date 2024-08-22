# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from datasets import (
    DownloadConfig,
    StreamingDownloadManager,
    get_dataset_config_names,
    get_dataset_default_config_name,
    load_dataset_builder,
)
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from datasets.exceptions import ConnectionError, DatasetNotFoundError, DefunctDatasetError, PermissionError
from datasets.exceptions import (
    DataFilesNotFoundError as _DataFilesNotFoundError,
)
from huggingface_hub.utils import HfHubHTTPError
from libcommon.exceptions import (
    ConfigNamesError,
    DataFilesNotFoundError,
    DatasetModuleNotInstalledError,
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
from worker.utils import resolve_trust_remote_code


def compute_config_names_response(
    dataset: str,
    max_number: int,
    dataset_scripts_allow_list: list[str],
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
        dataset_scripts_allow_list (`list[str]`):
            List of datasets for which we support dataset scripts.
            Unix shell-style wildcards also work in the dataset name for namespaced datasets,
            for example `some_namespace/*` to refer to all the datasets in the `some_namespace` namespace.
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)

    Raises:
        [~`libcommon.exceptions.EmptyDatasetError`]:
          The dataset is empty.
        [~`libcommon.exceptions.DatasetModuleNotInstalledError`]:
          The dataset tries to import a module that is not installed.
        [~`libcommon.exceptions.ConfigNamesError`]:
          If the list of configs could not be obtained using the datasets library.
        [~`libcommon.exceptions.DatasetWithScriptNotSupportedError`]:
            If the dataset has a dataset script and is not in the allow list.

    Returns:
        `DatasetConfigNamesResponse`: An object with the list of config names.
    """
    logging.info(f"compute 'dataset-config-names' for {dataset=}")
    # get the list of splits in streaming mode
    try:
        trust_remote_code = resolve_trust_remote_code(dataset=dataset, allow_list=dataset_scripts_allow_list)
        default_config_name: Optional[str] = None
        config_names = get_dataset_config_names(
            path=dataset,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )
        if len(config_names) > 1:
            default_config_name = get_dataset_default_config_name(
                path=dataset,
                token=hf_token,
                trust_remote_code=trust_remote_code,
            )
            # we might have to ignore defunct configs for datasets with a script
            if trust_remote_code:
                for config in list(config_names):
                    try:
                        builder = load_dataset_builder(
                            path=dataset, name=config, token=hf_token, trust_remote_code=trust_remote_code
                        )
                        dl_manager = StreamingDownloadManager(
                            download_config=DownloadConfig(token=hf_token),
                            base_path=builder.base_path,
                            dataset_name=builder.dataset_name,
                        )
                        # this raises DefunctDatasetError if the config is defunct
                        builder._split_generators(dl_manager)
                    except Exception as err:
                        if isinstance(err, DefunctDatasetError):
                            config_names.remove(config)
                            logging.info(f"Config {config} is defunct - ignoring it.")
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
    except ImportError as err:
        # this should only happen if the dataset is in the allow list, which should soon disappear
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

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_config_names_response(
                dataset=self.dataset,
                hf_token=self.app_config.common.hf_token,
                max_number=self.app_config.config_names.max_number,
                dataset_scripts_allow_list=self.app_config.common.dataset_scripts_allow_list,
            )
        )
