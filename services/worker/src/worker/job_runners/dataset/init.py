# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from collections.abc import Iterator
from typing import Literal, Optional, overload

from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from datasets.exceptions import (
    DataFilesNotFoundError as _DataFilesNotFoundError,
)
from datasets.exceptions import DatasetNotFoundError
from datasets.load import dataset_module_factory, get_dataset_builder_class
from datasets.packaged_modules.parquet.parquet import Parquet
from datasets.utils.py_utils import asdict
from huggingface_hub import HfFileSystem
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
from worker.job_runners.config.parquet_and_info import retry_get_features_num_examples_size_and_num_bytes, raise_if_long_column_name, backward_compat_features, ConfigParquetAndInfoJobRunner, ConfigParquetAndInfoResponse, SplitHubFile
from worker.job_runners.config.split_names import ConfigSplitNamesJobRunner, SplitsList, FullSplitItem
from worker.job_runners.config.parquet import ConfigParquetJobRunner, ConfigParquetResponse, SplitHubFile
from worker.job_runners.config.parquet_metadata import ConfigParquetMetadataJobRunner, ConfigParquetMetadataResponse, create_parquet_metadata_dir, StrPath, DATASET_SEPARATOR, ParquetFileMetadataItem


try:
    import libviewer as lv  # type: ignore
except ImportError:
    pass

def _missing_file(path: str) -> int:
    """Helper for get_file_sizes: raise when a cached file is not found."""
    raise KeyError(f"File not in dircache: '{path}'")


@overload
def get_file_sizes(
    fs: HfFileSystem,
    file_paths: list[str],
    *,
    ignore_missing: Literal[False] = False,
) -> dict[str, int]:
    pass


@overload
def get_file_sizes(
    fs: HfFileSystem,
    file_paths: list[str],
    *,
    ignore_missing: Literal[True],
) -> dict[str, int]:
    pass


def get_file_sizes(
    fs: HfFileSystem,
    file_paths: list[str],
    ignore_missing: bool = False,
) -> dict[str, int | None]:
    """
    Efficiently return file sizes for a list of files using the dircache.

    This method builds an in-memory path→size index from the dircache and looks up
    each requested file in O(1) time. It avoids any network calls and only uses
    already-cached directory listings.

    It assumes the HfFileSystem instance has all the info cached already.
    This is the case after instantiating a builder.

    Args:
        file_paths (`list[str]`):
            List of file paths (e.g. `["my-repo/file.txt", "my-repo/data/readme.md"]`).
        ignore_missing (`bool`, *optional*):
            If True, missing files map to `None` instead of raising. Defaults to False.

    Returns:
        `dict[str, int | None]`: Mapping from each file path to its size in bytes,
        or `None` if the file was not found in the cache (when `ignore_missing=True`).

    Example:
        ```python
        >>> # (Optional) clear the cache first
        >>> HfFileSystem.clear_instance_cache()
        >>> fs = HfFileSystem()
        >>> # Populate dircache
        >>> fs.ls("datasets/my-username/my-dataset", recursive=True)
        >>> # Or populate with a builder
        >>> # builder = load_dataset_builder(...)  # or builder = builder_cls(...)
        >>> get_file_sizes(fs, ["datasets/my-username/my-dataset/data.parquet"])
        {
            "datasets/my-username/my-dataset/data.parquet": 2500000000,
        }
        ```
    """
    # Build a path → size index from dircache in a single pass.
    # dircache: {parent_path: [{"name": full_path, "size": ..., ...}, ...]}
    size_index: dict[str, int] = {}
    for file_infos in fs.dircache.values():
        for info in file_infos:
            if info.get("type") == "file":
                size_index[info["name"]] = info.get("size", 0)

    return {path: size_index.get(path, None if ignore_missing else _missing_file(path)) for path in file_paths}


def compute_init_responses(
    dataset: str,
    max_num_configs: int,
    hf_endpoint: str,
    hf_token: Optional[str],
    committer_hf_token: Optional[str],
    source_revision: str,
    target_revision: str,
    commit_message: str,
    url_template: str,
    max_dataset_size_bytes: int,
    data_store: Optional[str],
    parquet_metadata_directory: StrPath,
    max_parallelism: int = 4,
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
    dataset_init_response: DatasetInitResponse = {"successes": [], "failed": []}
    HfFileSystem.clear_instance_cache()
    fs = HfFileSystem(endpoint=hf_endpoint, token=hf_token)
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
    builder_cls = get_dataset_builder_class(dataset_module)
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

    if not issubclass(builder_cls, Parquet):
        yield JobResult(dataset_init_response, progress=1.0)
    else:
        yield JobResult(dataset_init_response, progress=0.1)

        # config-split-names
        config_name_item = config_name_items[0]
        config = config_name_item["config"]
        logging.info(f"Loading {dataset=} config={config_name_item['config']} builder. ")
        builder = builder_cls(
            config_name=config,
            hash=dataset_module.hash,
            **dataset_module.builder_kwargs
        )
        job = {
            "dataset": dataset,
            "kind": ConfigSplitNamesJobRunner.get_job_type(),
            "config": config,
            "split": None,
        }
        dataset_init_response["successes"].append(job)
        split_items = [FullSplitItem(dataset=dataset, config=config, split=split) for split in builder.config.data_files]
        yield ShortcutJobResult(
            content=SplitsList(splits=split_items),
            job=job,
        )
        yield JobResult(dataset_init_response, progress=0.2)

        # config-parquet
        all_sizes = get_file_sizes(fs, file_paths=[fs._strip_protocol(data_file) for data_files in builder.config.data_files for data_file in data_files])
        job = {
            "dataset": dataset,
            "kind": ConfigParquetJobRunner.get_job_type(),
            "config": config,
            "split": None,
        }
        first_data_file = builder.config.data_files[split_items[0]["split"]][0]
        features, first_file_num_examples, first_file_size, first_file_num_bytes = retry_get_features_num_examples_size_and_num_bytes(fs.url(first_data_file), hf_endpoint=hf_endpoint, hf_token=hf_token)
        raise_if_long_column_name(features)
        dataset_init_response["successes"].append(job)
        parquet_file_items = [
            SplitHubFile(
                dataset=dataset,
                config=config,
                split=split,
                url=fs.url(data_file),
                filename=fs.resolve_path(data_file).path_in_repo,
                size=all_sizes[fs._strip_protocol(data_file)],
            )
            for split in builder.config.data_files
            for data_file in builder.config.data_files[split]
        ]
        yield ShortcutJobResult(
            content=ConfigParquetResponse(parquet_files=parquet_file_items, features=features, partial=False),
            job=job,
        )
        yield JobResult(dataset_init_response, progress=0.3)

        # config-parquet-and-info and config-parquet-metadata
        create_parquet_metadata_dir(
            dataset=dataset,
            config=config,
            split=split_items[0]["split"],
            parquet_metadata_directory=parquet_metadata_directory,
        )
        parquet_metadata_dir_subpath = f"{dataset}/{DATASET_SEPARATOR}"
        # todo: check path is right
        files = [
            {
                "path": fs.resolve_path(data_file).path_in_repo,
                "size": all_sizes[fs._strip_protocol(data_file)],
                "num_rows": None,
                "metadata_path": f"{parquet_metadata_dir_subpath}/{config}/{split}/{fs.resolve_path(data_file).path_in_repo}",
            }
            for split, data_files in builder.config.data_files.items()
            for data_file in data_files
        ]
        viewer = lv.Dataset(
            name=dataset,
            files=files,
            revision=fs.resolve_path(first_data_file).revision,
            hf_token=hf_token,
            hf_endpoint=hf_endpoint,
            data_store=data_store,
            metadata_store=f"file://{parquet_metadata_directory}",
        )
        result = viewer.sync_index(max_parallelism=max_parallelism)
        # todo: fill parquet_files_metadata correctly
        parquet_files_metadata: list[ParquetFileMetadataItem] = [
            {
                "dataset": item["dataset"],
                "config": item["config"],
                "split": item["split"],
                "url": item["url"],
                "filename": item["filename"],
                "size": item["size"],
                "num_rows": res["num_rows"],
                "parquet_metadata_subpath": res["metadata_path"],
            }
            for item, res in zip(parquet_file_items, result)
        ]
        # todo prepare parquet metadata response
        # todo: fill info
        dataset_info = asdict(builder.info)
        dataset_info["features"] = backward_compat_features(dataset_info["features"])
        job = {
            "dataset": dataset,
            "kind": ConfigParquetAndInfoJobRunner.get_job_type(),
            "config": config,
            "split": None,
        }
        dataset_init_response["successes"].append(job)
        split_items = [FullSplitItem(dataset=dataset, config=config, split=split) for split in builder.config.data_files]
        yield ShortcutJobResult(
            content=SplitsList(splits=split_items),
            job=job,
        )
        yield JobResult(dataset_init_response, progress=0.4)




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
