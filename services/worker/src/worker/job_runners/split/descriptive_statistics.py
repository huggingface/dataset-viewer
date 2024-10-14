# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Optional, TypedDict, Union

import polars as pl
import pyarrow.parquet as pq
from datasets import Features, Sequence
from datasets.features.features import FeatureType, _ArrayXD
from libcommon.dtos import JobInfo
from libcommon.exceptions import (
    CacheDirectoryNotInitializedError,
    FeaturesResponseEmptyError,
    NoSupportedFeaturesError,
    ParquetResponseEmptyError,
    PolarsParquetReadError,
    PreviousStepFormatError,
)
from libcommon.parquet_utils import (
    extract_split_directory_from_parquet_url,
    get_num_parquet_files_to_process,
    is_list_pa_type,
    parquet_export_is_partial,
)
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.storage import StrPath
from libcommon.utils import download_file_from_hub

from worker.config import AppConfig, DescriptiveStatisticsConfig
from worker.dtos import CompleteJobResult
from worker.job_runners.split.split_job_runner import SplitJobRunnerWithCache
from worker.statistics_utils import (
    FLOAT_DTYPES,
    INTEGER_DTYPES,
    NUMERICAL_DTYPES,
    STRING_DTYPES,
    AudioColumn,
    BoolColumn,
    ClassLabelColumn,
    DatetimeColumn,
    FloatColumn,
    ImageColumn,
    IntColumn,
    ListColumn,
    StatisticsPerColumnItem,
    StringColumn,
)

REPO_TYPE = "dataset"


class SplitDescriptiveStatisticsResponse(TypedDict):
    num_examples: int
    statistics: list[StatisticsPerColumnItem]
    partial: bool


SupportedColumns = Union[
    ClassLabelColumn,
    IntColumn,
    FloatColumn,
    StringColumn,
    BoolColumn,
    ListColumn,
    AudioColumn,
    ImageColumn,
    DatetimeColumn,
]


def is_extension_feature(feature: FeatureType) -> bool:
    """Check if a (possibly nested) feature is an arrow extension feature (Array2D, Array3D, Array4D, or Array5D)."""
    if isinstance(feature, dict):
        return any(is_extension_feature(f) for f in feature.values())
    elif isinstance(feature, (list, tuple)):
        return any(is_extension_feature(f) for f in feature)
    elif isinstance(feature, Sequence):
        return is_extension_feature(feature.feature)
    else:
        return isinstance(feature, _ArrayXD)


def get_extension_features(features: dict[str, Any]) -> set[str]:
    """Return set of names of extension features (Array2D, Array3D, Array4D, or Array5D) within provided features."""
    features = Features.from_dict(features)
    return {feature_name for feature_name, feature in features.items() if is_extension_feature(feature)}


def compute_descriptive_statistics_response(
    dataset: str,
    config: str,
    split: str,
    local_parquet_directory: Path,
    hf_token: Optional[str],
    parquet_revision: str,
    max_split_size_bytes: int,
    parquet_metadata_directory: StrPath,
) -> SplitDescriptiveStatisticsResponse:
    """
    Get the response of 'split-descriptive-statistics' for one specific split of a dataset from huggingface.co.
    Currently, integers, floats and ClassLabel features are supported.

    Args:
        dataset (`str`):
            Name of a dataset.
        config (`str`):
            Requested dataset configuration name.
        split (`str`):
            Requested dataset split.
        local_parquet_directory (`Path`):
            Path to a local directory where the dataset's parquet files are stored. We download these files locally
            because it enables fast querying and statistics computation.
        hf_token (`str`, *optional*):
            An app authentication token with read access to all the datasets.
        parquet_revision (`str`):
            The git revision (e.g. "refs/convert/parquet") from where to download the dataset's parquet files.
        max_split_size_bytes (`int`):
            If raw uncompressed split data is larger than this value, the statistics are computed
            only on the first parquet files, approximately up to this size, and the `partial` field will be set
            to `True` in the response.
        parquet_metadata_directory (`StrPath`):
            Path to directory on local shared storage containing parquet metadata files. Parquet metadata is needed
            to get uncompressed size of split files to determine the number of files to use if split is larger
            than `max_split_size_bytes`

    Raises:
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step does not have the expected format.
        [~`libcommon.exceptions.ParquetResponseEmptyError`]:
            If response for `config-parquet-and-info` doesn't have any parquet files.
        [~`libcommon.exceptions.FeaturesResponseEmptyError`]:
            If response for `config-parquet-and-info` doesn't have features.
        [~`libcommon.exceptions.NoSupportedFeaturesError`]:
            If requested dataset doesn't have any supported for statistics computation features.
            Currently, floats, integers and ClassLabels are supported.
        [~`libcommon.exceptions.StatisticsComputationError`]:
            If there was some unexpected behaviour during statistics computation.

    Returns:
        `SplitDescriptiveStatisticsResponse`: An object with the statistics response for a requested split, per each
            numerical (int and float) or ClassLabel feature.
    """

    logging.info(f"compute 'split-descriptive-statistics' for {dataset=} {config=} {split=}")

    # get parquet urls and dataset_info
    config_parquet_metadata_step = "config-parquet-metadata"
    parquet_metadata_response = get_previous_step_or_raise(
        kind=config_parquet_metadata_step,
        dataset=dataset,
        config=config,
    )
    content_parquet_metadata = parquet_metadata_response["content"]
    try:
        split_parquet_files = [
            parquet_file
            for parquet_file in content_parquet_metadata["parquet_files_metadata"]
            if parquet_file["config"] == config and parquet_file["split"] == split
        ]
        features = content_parquet_metadata["features"]

    except KeyError as e:
        raise PreviousStepFormatError(
            f"Previous step '{config_parquet_metadata_step}' did not return the expected content", e
        ) from e

    if not split_parquet_files:
        raise ParquetResponseEmptyError("No parquet files found.")

    if not features:
        raise FeaturesResponseEmptyError("No features found.")

    num_parquet_files_to_process, num_bytes, num_rows = get_num_parquet_files_to_process(
        parquet_files=split_parquet_files,
        parquet_metadata_directory=parquet_metadata_directory,
        max_size_bytes=max_split_size_bytes,
    )
    partial_parquet_export = parquet_export_is_partial(split_parquet_files[0]["url"])
    partial = partial_parquet_export or (num_parquet_files_to_process < len(split_parquet_files))
    split_parquet_files = split_parquet_files[:num_parquet_files_to_process]

    # store data as local parquet files for fast querying
    logging.info(f"Downloading remote parquet files to a local directory {local_parquet_directory}. ")
    # For directories like "partial-train" for the file at "en/partial-train/0000.parquet" in the C4 dataset.
    # Note that "-" is forbidden for split names so it doesn't create directory names collisions.
    split_directory = extract_split_directory_from_parquet_url(split_parquet_files[0]["url"])
    for parquet_file in split_parquet_files:
        download_file_from_hub(
            repo_type=REPO_TYPE,
            revision=parquet_revision,
            repo_id=dataset,
            filename=f"{config}/{split_directory}/{parquet_file['filename']}",
            local_dir=local_parquet_directory,
            hf_token=hf_token,
            cache_dir=local_parquet_directory,
            force_download=True,
            resume_download=False,
        )

    local_parquet_split_directory = Path(local_parquet_directory) / config / split_directory

    pq_split_dataset = pq.ParquetDataset(local_parquet_split_directory)
    num_examples = sum(fragment.metadata.num_rows for fragment in pq_split_dataset.fragments)
    split_extension_features = get_extension_features(features)
    features = {
        feature_name: feature
        for feature_name, feature in features.items()
        if feature_name not in split_extension_features
    }

    def _column_from_feature(
        dataset_feature_name: str, dataset_feature: Union[dict[str, Any], list[Any]]
    ) -> Optional[SupportedColumns]:
        if isinstance(dataset_feature, list) or (
            isinstance(dataset_feature, dict) and dataset_feature.get("_type") == "Sequence"
        ):
            # Compute only if it's internally a List! because it can also be Struct, see
            # https://huggingface.co/docs/datasets/v2.18.0/en/package_reference/main_classes#datasets.Features
            if is_list_pa_type(
                local_parquet_split_directory / split_parquet_files[0]["filename"], dataset_feature_name
            ):
                return ListColumn(feature_name=dataset_feature_name, n_samples=num_examples)

        if isinstance(dataset_feature, dict):
            _type = dataset_feature.get("_type")
            if _type == "ClassLabel":
                return ClassLabelColumn(
                    feature_name=dataset_feature_name, n_samples=num_examples, feature_dict=dataset_feature
                )

            if _type == "Audio":
                return AudioColumn(feature_name=dataset_feature_name, n_samples=num_examples)

            if _type == "Image":
                return ImageColumn(feature_name=dataset_feature_name, n_samples=num_examples)

            if _type == "Value":
                dtype = dataset_feature.get("dtype", "")
                if dtype in INTEGER_DTYPES:
                    return IntColumn(feature_name=dataset_feature_name, n_samples=num_examples)

                if dtype in FLOAT_DTYPES:
                    return FloatColumn(feature_name=dataset_feature_name, n_samples=num_examples)

                if dtype in STRING_DTYPES:
                    return StringColumn(feature_name=dataset_feature_name, n_samples=num_examples)

                if dtype == "bool":
                    return BoolColumn(feature_name=dataset_feature_name, n_samples=num_examples)

                if dtype.startswith("timestamp"):
                    return DatetimeColumn(feature_name=dataset_feature_name, n_samples=num_examples)
        return None

    columns: list[SupportedColumns] = []
    all_stats: list[StatisticsPerColumnItem] = []
    for feature_name, feature in features.items():
        if (column := _column_from_feature(feature_name, feature)) is not None:
            columns.append(column)

    if not columns:
        raise NoSupportedFeaturesError(
            "No columns for statistics computation found. Currently supported feature types are: "
            f"{NUMERICAL_DTYPES}, {STRING_DTYPES}, ClassLabel, Image, Audio, list/Sequence, datetime and bool. "
        )

    column_names_str = ", ".join([column.name for column in columns])
    column_counts = Counter([column.__class__.__name__ for column in columns])
    logging.info(
        f"Computing statistics for {len(columns)} columns: {column_names_str},"
        f"\nColumn types counts: {column_counts}. "
    )

    for column in columns:
        if isinstance(column, AudioColumn) or isinstance(column, ImageColumn):
            column_stats = column.compute_and_prepare_response(local_parquet_split_directory)
        else:
            try:
                if split_extension_features:
                    data = pl.DataFrame._from_arrow(
                        pq.read_table(local_parquet_split_directory, columns=[column.name])
                    )
                else:
                    data = pl.read_parquet(local_parquet_split_directory / "*.parquet", columns=[column.name])
            except Exception as error:
                raise PolarsParquetReadError(
                    f"Error reading parquet file(s) at {local_parquet_split_directory=}, columns=[{column.name}]: {error}",
                    error,
                )
            column_stats = column.compute_and_prepare_response(data)
        all_stats.append(column_stats)

    if not all_stats:
        raise NoSupportedFeaturesError(
            "No columns for statistics computation found. Currently supported feature types are: "
            f"{NUMERICAL_DTYPES}, {STRING_DTYPES}, ClassLabel, list/Sequence and bool. "
        )

    logging.info(f"Computing for {dataset=} {config=} {split=} finished. {len(all_stats)} columns processed. ")

    return SplitDescriptiveStatisticsResponse(
        num_examples=num_examples,
        statistics=sorted(all_stats, key=lambda x: x["column_name"]),
        partial=partial,
    )


class SplitDescriptiveStatisticsJobRunner(SplitJobRunnerWithCache):
    descriptive_statistics_config: DescriptiveStatisticsConfig

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        statistics_cache_directory: StrPath,
        parquet_metadata_directory: StrPath,
    ):
        super().__init__(
            job_info=job_info,
            app_config=app_config,
            cache_directory=Path(statistics_cache_directory),
        )
        self.descriptive_statistics_config = app_config.descriptive_statistics
        self.parquet_metadata_directory = parquet_metadata_directory

    @staticmethod
    def get_job_type() -> str:
        return "split-descriptive-statistics"

    def compute(self) -> CompleteJobResult:
        if self.cache_subdirectory is None:
            raise CacheDirectoryNotInitializedError("Cache directory has not been initialized.")
        return CompleteJobResult(
            compute_descriptive_statistics_response(
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                local_parquet_directory=self.cache_subdirectory,
                hf_token=self.app_config.common.hf_token,
                parquet_revision=self.descriptive_statistics_config.parquet_revision,
                max_split_size_bytes=self.descriptive_statistics_config.max_split_size_bytes,
                parquet_metadata_directory=self.parquet_metadata_directory,
            )
        )
