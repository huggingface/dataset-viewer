# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from typing import Optional

from datasets import get_dataset_split_names
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from libcommon.dtos import FullSplitItem
from libcommon.exceptions import (
    DatasetWithScriptNotSupportedError,
    DatasetWithTooManySplitsError,
    EmptyDatasetError,
    PreviousStepFormatError,
    SplitNamesFromStreamingError,
)
from libcommon.simple_cache import CachedArtifactError, CachedArtifactNotFoundError, get_previous_step_or_raise

from worker.dtos import CompleteJobResult, SplitsList
from worker.job_runners.config.config_job_runner import ConfigJobRunnerWithDatasetsCache


def compute_split_names_from_streaming_response(
    dataset: str,
    config: str,
    max_number: int,
    hf_token: Optional[str] = None,
) -> SplitsList:
    """
    Get the response of 'config-split-names' for one specific dataset and config on huggingface.co, using streaming.

    This function relies on the streaming mode if the splits are not directly defined in the dataset config. See
    https://github.dev/huggingface/datasets/blob/e183a269067575db8765ee979bd8523d14a1adae/src/datasets/inspect.py#L389-L390

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        max_number (`str`):
            Maximum number of splits.

    Raises:
        [~`libcommon.exceptions.EmptyDatasetError`]:
          The dataset is empty.
        [~`libcommon.exceptions.SplitsNamesError`]:
          If the list of splits could not be obtained using the datasets library.
        [~`libcommon.exceptions.DatasetWithScriptNotSupportedError`]:
            If the dataset has a dataset script.
        [~`libcommon.exceptions.SplitNamesFromStreamingError`]:
            If the split names could not be obtained using the datasets library.
        [~`libcommon.exceptions.DatasetWithTooManySplitsError`]:
            If the config has too many splits.

    Returns:
        `SplitsList`: An object with the list of split names for the dataset and config.
    """
    logging.info(f"compute 'config-split-names' using streaming for {dataset=} {config=}")
    try:
        split_name_items: list[FullSplitItem] = [
            {"dataset": dataset, "config": config, "split": str(split)}
            for split in get_dataset_split_names(
                path=dataset,
                config_name=config,
                token=hf_token,
            )
        ]
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except Exception as err:
        if isinstance(err, ValueError) and "trust_remote_code" in str(err):
            raise DatasetWithScriptNotSupportedError from err
        raise SplitNamesFromStreamingError(
            f"Cannot get the split names for the config '{config}' of the dataset.",
            cause=err,
        ) from err
    if len(split_name_items) > max_number:
        split_examples = ", ".join([split_name_item["split"] for split_name_item in split_name_items[:5]])
        raise DatasetWithTooManySplitsError(
            f"The {config} config contains {len(split_name_items)} while it should generally contain 3 splits maximum (train/validation/test). "
            f"If the splits {split_examples}... are not used to differentiate between training and evaluation, please consider defining configs of this dataset instead. "
            "You can find how to define configs instead of splits here: https://huggingface.co/docs/hub/datasets-data-files-configuration"
        )
    return SplitsList(splits=split_name_items)


def compute_split_names_from_info_response(dataset: str, config: str, max_number: int) -> SplitsList:
    """
    Get the response of 'config-split-names' for one specific dataset and config on huggingface.co
    computed from cached response in dataset-info step.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
        max_number (`str`):
            Maximum number of splits.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.simple_cache.CachedArtifactNotFoundError`]:
          If the previous step has not been computed yet.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
          If the content of the previous step has not the expected format
        [~`libcommon.exceptions.DatasetWithTooManySplitsError`]:
            If the config has too many splits.

    Returns:
        `SplitsList`: An object with the list of split names for the dataset and config.
    """
    logging.info(f"compute 'config-split-names' from config-info for {dataset=} {config=}")
    config_info_response = get_previous_step_or_raise(kind="config-info", dataset=dataset, config=config)

    try:
        splits_content = config_info_response["content"]["dataset_info"]["splits"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step 'config-info' did not return the expected content.") from e

    split_name_items: list[FullSplitItem] = [
        {"dataset": dataset, "config": config, "split": str(split)} for split in splits_content
    ]
    if len(split_name_items) > max_number:
        split_examples = ", ".join([split_name_item["split"] for split_name_item in split_name_items[:5]])
        raise DatasetWithTooManySplitsError(
            f"The {config} config contains {len(split_name_items)} while it should generally contain 3 splits maximum (train/validation/test). "
            f"If the splits {split_examples}... are not used to differentiate between training and evaluation, please consider defining configs of this dataset instead. "
            "You can find how to define configs instead of splits here: https://huggingface.co/docs/hub/datasets-data-files-configuration"
        )
    return SplitsList(splits=split_name_items)


class ConfigSplitNamesJobRunner(ConfigJobRunnerWithDatasetsCache):
    @staticmethod
    def get_job_type() -> str:
        return "config-split-names"

    def compute(self) -> CompleteJobResult:
        try:
            return CompleteJobResult(
                compute_split_names_from_info_response(
                    dataset=self.dataset, config=self.config, max_number=self.app_config.split_names.max_number
                )
            )
        except (CachedArtifactError, CachedArtifactNotFoundError):
            logging.info(
                f"Cannot compute 'config-split-names' from config-info for {self.dataset=} {self.config=}. "
                f"Trying to compute it using streaming."
            )
            pass
        return CompleteJobResult(
            compute_split_names_from_streaming_response(
                dataset=self.dataset,
                config=self.config,
                max_number=self.app_config.split_names.max_number,
                hf_token=self.app_config.common.hf_token,
            )
        )
