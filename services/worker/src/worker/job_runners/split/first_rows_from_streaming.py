# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from pathlib import Path
from typing import Optional

from datasets import IterableDataset, get_dataset_config_info, load_dataset
from libcommon.constants import MAX_NUM_ROWS_PER_PAGE
from libcommon.dtos import JobInfo, RowsContent, SplitFirstRowsResponse
from libcommon.exceptions import (
    DatasetWithScriptNotSupportedError,
    FeaturesError,
    InfoError,
)
from libcommon.storage_client import StorageClient
from libcommon.viewer_utils.rows import create_first_rows_response

from worker.config import AppConfig, FirstRowsConfig
from worker.dtos import CompleteJobResult
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithDatasetsCache
from worker.utils import get_rows_or_raise, resolve_trust_remote_code


def compute_first_rows_response(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    storage_client: StorageClient,
    hf_token: Optional[str],
    min_cell_bytes: int,
    rows_max_bytes: int,
    rows_max_number: int,
    rows_min_number: int,
    columns_max_number: int,
    dataset_scripts_allow_list: list[str],
    max_size_fallback: Optional[int] = None,
) -> SplitFirstRowsResponse:
    """
    Get the response of 'first-rows-from-streaming' for one specific split of a dataset from huggingface.co.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        revision (`str`):
            The git revision of the dataset.
        config (`str`):
            A configuration name.
        split (`str`):
            A split name.
        storage_client (`StorageClient`):
            A storage client to save the assets (images, audio, etc.).
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
        min_cell_bytes (`int`):
            The minimum number of bytes for a cell, when truncation applies.
        rows_max_bytes (`int`):
            The maximum number of bytes of the response (else, the response is truncated).
        rows_max_number (`int`):
            The maximum number of rows of the response.
        rows_min_number (`int`):
            The minimum number of rows of the response.
        columns_max_number (`int`):
            The maximum number of columns supported.
        dataset_scripts_allow_list (`list[str]`):
            List of datasets for which we support dataset scripts.
            Unix shell-style wildcards also work in the dataset name for namespaced datasets,
            for example `some_namespace/*` to refer to all the datasets in the `some_namespace` namespace.
            The keyword `{{ALL_DATASETS_WITH_NO_NAMESPACE}}` refers to all the datasets without namespace.
        max_size_fallback (`int`, *optional*): **DEPRECATED**
            The maximum number of bytes of the split to fallback to normal mode if the streaming mode fails.
            This argument is now hard-coded to 100MB, and will be removed in a future version.

    Raises:
        [~`libcommon.exceptions.SplitNotFoundError`]:
          If the split does not exist in the dataset.
        [~`libcommon.exceptions.InfoError`]:
          If the config info could not be obtained using the datasets library.
        [~`libcommon.exceptions.FeaturesError`]:
          If the split features could not be obtained using the datasets library.
        [~`libcommon.exceptions.RowsPostProcessingError`]:
          If the post-processing of the split rows failed, e.g. while saving the images or audio files to the assets.
        [~`libcommon.exceptions.TooManyColumnsError`]:
          If the number of columns (features) exceeds the maximum supported number of columns.
        [~`libcommon.exceptions.TooBigContentError`]:
          If the first rows content exceeds the maximum supported size of bytes.
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
          If the content of the previous step has not the expected format
        [~`libcommon.exceptions.StreamingRowsError`]:
          If the split rows could not be obtained using the datasets library in streaming mode.
        [~`libcommon.exceptions.NormalRowsError`]:
          If the split rows could not be obtained using the datasets library in normal mode.
        [~`libcommon.exceptions.DatasetWithScriptNotSupportedError`]:
            If the dataset has a dataset script and is not in the allow list.

    Returns:
        `SplitFirstRowsResponse`: The list of first rows of the split.
    """
    logging.info(f"get 'first-rows-from-streaming' for {dataset=} {config=} {split=}")
    trust_remote_code = resolve_trust_remote_code(dataset=dataset, allow_list=dataset_scripts_allow_list)
    # get the features
    try:
        info = get_dataset_config_info(
            path=dataset, config_name=config, token=hf_token, trust_remote_code=trust_remote_code
        )
    except Exception as err:
        if isinstance(err, ValueError) and "trust_remote_code" in str(err):
            raise DatasetWithScriptNotSupportedError(
                "The dataset viewer doesn't support this dataset because it runs "
                "arbitrary python code. Please open a discussion in the discussion tab "
                "if you think this is an error and tag @lhoestq and @severo."
            ) from err
        raise InfoError(
            f"The info cannot be fetched for the config '{config}' of the dataset.",
            cause=err,
        ) from err
    if not info.features:
        try:
            # https://github.com/huggingface/datasets/blob/f5826eff9b06ab10dba1adfa52543341ef1e6009/src/datasets/iterable_dataset.py#L1255
            iterable_dataset = load_dataset(
                path=dataset,
                name=config,
                split=split,
                streaming=True,
                token=hf_token,
                trust_remote_code=trust_remote_code,
            )
            if not isinstance(iterable_dataset, IterableDataset):
                raise TypeError("load_dataset should return an IterableDataset.")
            iterable_dataset = iterable_dataset._resolve_features()
            if not isinstance(iterable_dataset, IterableDataset):
                raise TypeError("load_dataset should return an IterableDataset.")
            features = iterable_dataset.features
        except Exception as err:
            if isinstance(err, ValueError) and "trust_remote_code" in str(err):
                raise DatasetWithScriptNotSupportedError(
                    "The dataset viewer doesn't support this dataset because it runs "
                    "arbitrary python code. Please open a discussion in the discussion tab "
                    "if you think this is an error and tag @lhoestq and @severo."
                ) from err
            raise FeaturesError(
                (
                    f"Cannot extract the features (columns) for the split '{split}' of the config '{config}' of the"
                    " dataset."
                ),
                cause=err,
            ) from err
    else:
        features = info.features

    def get_rows_content(rows_max_number: int) -> RowsContent:
        return get_rows_or_raise(
            dataset=dataset,
            config=config,
            split=split,
            info=info,
            max_size_fallback=max_size_fallback,
            rows_max_number=rows_max_number,
            token=hf_token,
            trust_remote_code=trust_remote_code,
        )

    return create_first_rows_response(
        dataset=dataset,
        revision=revision,
        config=config,
        split=split,
        storage_client=storage_client,
        features=features,
        get_rows_content=get_rows_content,
        min_cell_bytes=min_cell_bytes,
        rows_max_bytes=rows_max_bytes,
        rows_max_number=rows_max_number,
        rows_min_number=rows_min_number,
        columns_max_number=columns_max_number,
    )


class SplitFirstRowsFromStreamingJobRunner(SplitJobRunnerWithDatasetsCache):
    first_rows_config: FirstRowsConfig

    @staticmethod
    def get_job_type() -> str:
        return "split-first-rows-from-streaming"

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        hf_datasets_cache: Path,
        storage_client: StorageClient,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.first_rows_config = app_config.first_rows
        self.storage_client = storage_client

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_first_rows_response(
                dataset=self.dataset,
                revision=self.dataset_git_revision,
                config=self.config,
                split=self.split,
                storage_client=self.storage_client,
                hf_token=self.app_config.common.hf_token,
                min_cell_bytes=self.first_rows_config.min_cell_bytes,
                rows_max_bytes=self.first_rows_config.max_bytes,
                rows_min_number=self.first_rows_config.min_number,
                rows_max_number=MAX_NUM_ROWS_PER_PAGE,
                columns_max_number=self.first_rows_config.columns_max_number,
                dataset_scripts_allow_list=self.app_config.common.dataset_scripts_allow_list,
            )
        )
