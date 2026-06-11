# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from collections.abc import Iterator
from typing import Optional

from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from datasets.exceptions import (
    DataFilesNotFoundError as _DataFilesNotFoundError,
)
from datasets.exceptions import DatasetNotFoundError
from datasets.load import dataset_module_factory, get_dataset_builder_class
from huggingface_hub.utils import HfHubHTTPError
from libcommon.dtos import CachedJob
from libcommon.exceptions import (
    ConfigNamesError,
    DataFilesNotFoundError,
    DatasetWithScriptNotSupportedError,
    DatasetWithTooManyConfigsError,
    EmptyDatasetError,
    FileFormatMismatchBetweenSplitsError,
    RetryableConfigNamesError,
)

from worker.dtos import (
    ConfigNameItem,
    DatasetConfigNamesResponse,
    DatasetInitResponse,
    JobResult,
    ShortcutJobResult,
)
from worker.job_runners.dataset.config_names import DatasetConfigNamesJobRunner
from worker.job_runners.dataset.dataset_job_runner import (
    DatasetJobRunnerWithDatasetsCache,
)
from worker.utils import resolve_hf_path


def compute_init_responses(
    dataset: str,
    max_num_configs: int,
    hf_token: Optional[str] = None,
) -> Iterator[JobResult]:
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
    logging.info(f"compute 'dataset-init' for {dataset=}")
    repo_dir = f"hf://datasets/{dataset}"
    dataset_init_response: DatasetInitResponse = {"successes": [], "failed": []}
    try:
        dataset_module = dataset_module_factory(dataset, token=hf_token)
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

    default_config_name: Optional[str] = None
    repo_dir_with_commit_hash = repo_dir + f"@{dataset_module.hash}"
    builder_cls = get_dataset_builder_class(dataset_module)

    # Safety checks
    for builder_config in builder_cls.builder_configs.values():
        data_files = builder_config.data_files
        if data_files is not None:
            for split in data_files:
                for data_file in data_files[split]:
                    resolved_data_file = resolve_hf_path(data_file)
                    if not resolved_data_file.startswith(repo_dir_with_commit_hash + "/"):
                        raise ValueError(f"Data files don't belong to {repo_dir}")

    config_names = list(builder_cls.builder_configs.keys())
    if "config_name" in dataset_module.builder_kwargs and isinstance(
        dataset_module.builder_kwargs["config_name"], str
    ):
        default_config_name = dataset_module.builder_kwargs["config_name"]
    elif builder_cls.DEFAULT_CONFIG_NAME:
        default_config_name = builder_cls.DEFAULT_CONFIG_NAME
    elif config_names:
        default_config_name = config_names[0] if len(config_names) == 1 else None
    else:
        default_config_name = "default"

    config_name_items: list[ConfigNameItem] = [
        {"dataset": dataset, "config": str(config)}
        for config in sorted(
            config_names,
            key=lambda config_name: (config_name != default_config_name, config_name),  # default config first
        )
    ]

    number_of_configs = len(config_name_items)
    if number_of_configs > max_num_configs:
        raise DatasetWithTooManyConfigsError(
            f"The maximum number of configs allowed is {max_num_configs}, dataset has {number_of_configs} configs."
        )

    job: CachedJob = {
        "dataset": dataset,
        "kind": DatasetConfigNamesJobRunner.get_job_type(),
        "config": None,
        "split": None,
    }
    dataset_init_response["successes"].append(job)
    yield ShortcutJobResult(
        content=DatasetConfigNamesResponse(config_names=config_name_items),
        job=job,
    )
    yield JobResult(dataset_init_response, progress=1.0)


class DatasetInitJobRunner(DatasetJobRunnerWithDatasetsCache):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-init"

    def compute(self) -> Iterator[JobResult]:
        yield from compute_init_responses(
            dataset=self.dataset,
            hf_token=self.app_config.common.hf_token,
            max_num_configs=self.app_config.config_names.max_number_for_init,
        )
