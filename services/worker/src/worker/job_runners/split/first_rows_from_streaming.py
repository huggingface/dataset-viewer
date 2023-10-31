# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from pathlib import Path
from typing import Optional

from datasets import (
    Audio,
    Features,
    Image,
    IterableDataset,
    get_dataset_config_info,
    load_dataset,
)
from libcommon.constants import (
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
    PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION,
)
from libcommon.exceptions import (
    FeaturesError,
    InfoError,
    RowsPostProcessingError,
    TooBigContentError,
    TooManyColumnsError,
)
from libcommon.processing_graph import ProcessingStep
from libcommon.public_assets_storage import PublicAssetsStorage
from libcommon.storage_client import StorageClient
from libcommon.utils import JobInfo, Row
from libcommon.viewer_utils.features import get_cell_value, to_features_list

from worker.config import AppConfig, FirstRowsConfig
from worker.dtos import CompleteJobResult, JobRunnerInfo, SplitFirstRowsResponse
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithDatasetsCache
from worker.utils import (
    create_truncated_row_items,
    disable_dataset_scripts_support,
    get_json_size,
    get_rows_or_raise,
)


def transform_rows(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    rows: list[Row],
    features: Features,
    public_assets_storage: PublicAssetsStorage,
) -> list[Row]:
    return [
        {
            featureName: get_cell_value(
                dataset=dataset,
                revision=revision,
                config=config,
                split=split,
                row_idx=row_idx,
                cell=row[featureName] if featureName in row else None,
                featureName=featureName,
                fieldType=fieldType,
                public_assets_storage=public_assets_storage,
            )
            for (featureName, fieldType) in features.items()
        }
        for row_idx, row in enumerate(rows)
    ]


def compute_first_rows_response(
    dataset: str,
    revision: str,
    config: str,
    split: str,
    public_assets_storage: PublicAssetsStorage,
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
    Get the response of /first-rows for one specific split of a dataset from huggingface.co.
    Dataset can be private or gated if you pass an acceptable token.

    It is assumed that the dataset exist and can be accessed using the token.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.
        split (`str`):
            A split name.
        public_assets_storage (`PublicAssetsStorage`):
            The public assets storage configuration.
        hf_endpoint (`str`):
            The Hub endpoint (for example: "https://huggingface.co")
        hf_token (`str` or `None`):
            An authentication token (See https://huggingface.co/settings/token)
        max_size_fallback (`int` or `None`): **DEPRECATED**
            The maximum number of bytes of the split to fallback to normal mode if the streaming mode fails.
            This argument is now hard-coded to 100MB, and will be removed in a future version.
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
    Returns:
        [`SplitFirstRowsResponse`]: The list of first rows of the split.
    Raises the following errors:
        - [`libcommon.exceptions.SplitNotFoundError`]
          If the split does not exist in the dataset.
        - [`libcommon.exceptions.InfoError`]
          If the config info could not be obtained using the datasets library.
        - [`libcommon.exceptions.FeaturesError`]
          If the split features could not be obtained using the datasets library.
        - [`libcommon.exceptions.RowsPostProcessingError`]
          If the post-processing of the split rows failed, e.g. while saving the images or audio files to the assets.
        - [`libcommon.exceptions.TooManyColumnsError`]
          If the number of columns (features) exceeds the maximum supported number of columns.
        - [`libcommon.exceptions.TooBigContentError`]
          If the first rows content exceeds the maximum supported size of bytes.
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
          If the content of the previous step has not the expected format
        - [`libcommon.exceptions.StreamingRowsError`]
          If the split rows could not be obtained using the datasets library in streaming mode.
        - [`libcommon.exceptions.NormalRowsError`]
          If the split rows could not be obtained using the datasets library in normal mode.
        - [`libcommon.exceptions.DatasetWithScriptNotSupportedError`]
            If the dataset has a dataset script and is not in the allow list.
    """
    logging.info(f"get first-rows for dataset={dataset} config={config} split={split}")
    # get the features
    try:
        with disable_dataset_scripts_support(dataset_scripts_allow_list):
            info = get_dataset_config_info(
                path=dataset,
                config_name=config,
                token=hf_token,
            )
    except Exception as err:
        raise InfoError(
            f"The info cannot be fetched for the config '{config}' of the dataset.",
            cause=err,
        ) from err
    if not info.features:
        try:
            # https://github.com/huggingface/datasets/blob/f5826eff9b06ab10dba1adfa52543341ef1e6009/src/datasets/iterable_dataset.py#L1255
            with disable_dataset_scripts_support(dataset_scripts_allow_list):
                iterable_dataset = load_dataset(
                    path=dataset,
                    name=config,
                    split=split,
                    streaming=True,
                    token=hf_token,
                )
            if not isinstance(iterable_dataset, IterableDataset):
                raise TypeError("load_dataset should return an IterableDataset.")
            iterable_dataset = iterable_dataset._resolve_features()
            if not isinstance(iterable_dataset, IterableDataset):
                raise TypeError("load_dataset should return an IterableDataset.")
            features = iterable_dataset.features
        except Exception as err:
            raise FeaturesError(
                (
                    f"Cannot extract the features (columns) for the split '{split}' of the config '{config}' of the"
                    " dataset."
                ),
                cause=err,
            ) from err
    else:
        features = info.features

    if features and len(features) > columns_max_number:
        raise TooManyColumnsError(
            f"The number of columns ({len(features)}) exceeds the maximum supported number of columns"
            f" ({columns_max_number}). This is a current limitation of the datasets viewer. You can reduce the number"
            " of columns if you want the viewer to work."
        )

    # validate size of response without the rows
    features_list = to_features_list(features=features)
    response_features_only: SplitFirstRowsResponse = {
        "dataset": dataset,
        "config": config,
        "split": split,
        "features": features_list,
        "rows": [],
        "truncated": False,
    }

    surrounding_json_size = get_json_size(response_features_only)
    if surrounding_json_size > rows_max_bytes:
        raise TooBigContentError(
            f"The size of the content of the first rows ({surrounding_json_size} B) exceeds the maximum"
            f" supported size ({rows_max_bytes} B) even after truncation. Please report the issue."
        )

    # get the rows
    with disable_dataset_scripts_support(dataset_scripts_allow_list):
        rows_content = get_rows_or_raise(
            dataset=dataset,
            config=config,
            split=split,
            info=info,
            max_size_fallback=max_size_fallback,
            rows_max_number=rows_max_number,
            token=hf_token,
        )
    rows = rows_content["rows"]
    all_fetched = rows_content["all_fetched"]

    # transform the rows, if needed (e.g. save the images or audio to the assets, and return their URL)
    try:
        transformed_rows = transform_rows(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            rows=rows,
            features=features,
            public_assets_storage=public_assets_storage,
        )
    except Exception as err:
        raise RowsPostProcessingError(
            "Server error while post-processing the split rows. Please report the issue.",
            cause=err,
        ) from err

    # truncate the rows to fit within the restrictions, and prepare them as RowItems
    columns_to_keep_untruncated = [col for col, feature in features.items() if isinstance(feature, (Image, Audio))]
    row_items, truncated = create_truncated_row_items(
        rows=transformed_rows,
        min_cell_bytes=min_cell_bytes,
        rows_max_bytes=rows_max_bytes - surrounding_json_size,
        rows_min_number=rows_min_number,
        columns_to_keep_untruncated=columns_to_keep_untruncated,
    )

    response = response_features_only
    response["rows"] = row_items
    response["truncated"] = (not all_fetched) or truncated

    # return the response
    return response


class SplitFirstRowsFromStreamingJobRunner(SplitJobRunnerWithDatasetsCache):
    first_rows_config: FirstRowsConfig

    @staticmethod
    def get_job_type() -> str:
        return "split-first-rows-from-streaming"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_STREAMING_VERSION

    @staticmethod
    def get_parallel_job_runner() -> JobRunnerInfo:
        return JobRunnerInfo(
            job_runner_version=PROCESSING_STEP_SPLIT_FIRST_ROWS_FROM_PARQUET_VERSION,
            job_type="split-first-rows-from-parquet",
        )

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        processing_step: ProcessingStep,
        hf_datasets_cache: Path,
        storage_client: StorageClient,
    ) -> None:
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            processing_step=processing_step,
            hf_datasets_cache=hf_datasets_cache,
        )
        self.first_rows_config = app_config.first_rows
        self.assets_base_url = app_config.assets.base_url
        self.public_assets_storage = PublicAssetsStorage(
            assets_base_url=self.assets_base_url,
            overwrite=True,
            storage_client=storage_client,
        )

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_first_rows_response(
                dataset=self.dataset,
                revision=self.dataset_git_revision,
                config=self.config,
                split=self.split,
                public_assets_storage=self.public_assets_storage,
                hf_token=self.app_config.common.hf_token,
                min_cell_bytes=self.first_rows_config.min_cell_bytes,
                rows_max_bytes=self.first_rows_config.max_bytes,
                rows_max_number=self.first_rows_config.max_number,
                rows_min_number=self.first_rows_config.min_number,
                columns_max_number=self.first_rows_config.columns_max_number,
                dataset_scripts_allow_list=self.app_config.common.dataset_scripts_allow_list,
            )
        )
