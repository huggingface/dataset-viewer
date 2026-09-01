# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import posixpath
from collections.abc import Iterator
from pathlib import Path
from typing import Literal, Optional, overload
from unittest.mock import patch

from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from datasets.exceptions import (
    DataFilesNotFoundError as _DataFilesNotFoundError,
)
from datasets.exceptions import DatasetNotFoundError
from datasets.load import dataset_module_factory, get_dataset_builder_class
from datasets.packaged_modules.parquet.parquet import Parquet
from datasets.utils.file_utils import is_relative_path
from datasets.utils.py_utils import asdict
from huggingface_hub import HfFileSystem
from huggingface_hub.utils import HfHubHTTPError
from libcommon.dtos import CachedJob, JobInfo, SplitFirstRowsResponse, SplitHubFile
from libcommon.exceptions import (
    ConfigNamesError,
    DataFilesNotFoundError,
    DatasetWithScriptNotSupportedError,
    DatasetWithTooManyConfigsError,
    EmptyDatasetError,
    FileFormatMismatchBetweenSplitsError,
    RetryableConfigNamesError,
)
from libcommon.storage import StrPath
from libcommon.storage_client import StorageClient

from worker.config import AppConfig
from worker.dtos import (
    ConfigInfoResponse,
    ConfigNameItem,
    ConfigParquetAndInfoResponse,
    ConfigParquetMetadataResponse,
    ConfigSize,
    ConfigSizeContent,
    ConfigSizeResponse,
    DatasetConfigNamesResponse,
    DatasetInfoResponse,
    DatasetInitResponse,
    DatasetSize,
    DatasetSizeContent,
    DatasetSizeResponse,
    IsValidResponse,
    JobResult,
    ParquetFileMetadataItem,
    ShortcutCompleteJobResult,
    ShortcutJobResult,
    SplitSize,
)
from worker.job_runners.config.info import ConfigInfoJobRunner
from worker.job_runners.config.is_valid import ConfigIsValidJobRunner
from worker.job_runners.config.parquet import ConfigParquetJobRunner, ConfigParquetResponse
from worker.job_runners.config.parquet_and_info import (
    ConfigParquetAndInfoJobRunner,
    backward_compat_features,
    fill_builder_info,
    http_backoff_with_timeout,
    raise_if_long_column_name,
    retry_get_features_num_examples_size_and_num_bytes,
)
from worker.job_runners.config.parquet_metadata import (
    DATASET_SEPARATOR,
    ConfigParquetMetadataJobRunner,
    create_parquet_metadata_dir,
)
from worker.job_runners.config.size import ConfigSizeJobRunner
from worker.job_runners.config.split_names import ConfigSplitNamesJobRunner, FullSplitItem, SplitsList
from worker.job_runners.dataset.config_names import DatasetConfigNamesJobRunner
from worker.job_runners.dataset.dataset_job_runner import (
    DatasetJobRunnerWithDatasetsCache,
)
from worker.job_runners.dataset.info import DatasetInfoJobRunner
from worker.job_runners.dataset.is_valid import DatasetIsValidJobRunner
from worker.job_runners.dataset.size import DatasetSizeJobRunner
from worker.job_runners.split.first_rows import SplitFirstRowsJobRunner, compute_first_rows_from_parquet_response
from worker.job_runners.split.is_valid import SplitIsValidJobRunner
from worker.utils import resolve_hf_path

try:
    import libviewer as lv  # type: ignore
except ImportError:
    pass


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
) -> dict[str, int]:
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
        `dict[str, int]`: Mapping from each file path to its size in bytes.
        Raises `KeyError` if a file was not found in the cache.

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
    remaining_files_to_scan = set(file_paths)
    file_sizes: dict[str, int] = {}
    for file_infos in fs.dircache.values():
        for info in file_infos:
            if info.get("type") == "file":
                path = info["name"]
                if path in remaining_files_to_scan:
                    file_sizes[path] = info.get("size", 0)
                    remaining_files_to_scan.remove(path)
    if remaining_files_to_scan:
        raise KeyError(f"Files not in dircache: {list(remaining_files_to_scan)}")

    return file_sizes


def compute_init_responses(
    dataset: str,
    max_num_configs: int,
    hf_endpoint: str,
    hf_token: Optional[str],
    data_store: Optional[str],
    parquet_metadata_directory: StrPath,
    max_parallelism: int = 4,
    hf_datasets_cache: Optional[Path] = None,
    storage_client: Optional[StorageClient] = None,
    rows_index_max_arrow_data_in_memory: int = 50_000_000,
    first_rows_columns_max_number: int = 100,
    first_rows_max_bytes: int = 200_000,
    first_rows_min_cell_bytes: int = 100,
    first_rows_min_number: int = 0,
    first_rows_max_number: int = 3,
) -> Iterator[JobResult]:
    """
    Get the response of 'dataset-init' for one specific dataset on huggingface.co.
    Dataset can be gated if you pass an acceptable token.
    It is assumed that the dataset exists and can be accessed using the token.

    Then, it tries to shortcut other jobs like `config-parquet-and-info` and `config-parquet-metadata`.

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
    repo_dir_with_commit_hash = repo_dir + f"@{dataset_module.hash}"
    builder_cls = get_dataset_builder_class(dataset_module)

    # Safety checks
    for builder_config in builder_cls.builder_configs.values():
        data_files = builder_config.data_files
        if data_files is not None:
            for split in data_files:
                for data_file in data_files[split]:
                    resolved_data_file = resolve_hf_path(
                        posixpath.join(dataset_module.builder_kwargs["base_path"], data_file)
                        if is_relative_path(data_file)
                        else data_file
                    )
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
    yield ShortcutCompleteJobResult(
        content=DatasetConfigNamesResponse(config_names=config_name_items),
        job=job,
    )

    if not issubclass(builder_cls, Parquet):
        # no shortcut, the other jobs run later
        yield JobResult(dataset_init_response, progress=1.0)
    else:
        # we can do shortcuts and get the other jobs results right now !

        yield JobResult(dataset_init_response, progress=0.1)

        # config-split-names
        config_name_item = config_name_items[0]
        config = config_name_item["config"]
        logging.info(f"Loading {dataset=} config={config_name_item['config']} builder. ")
        builder = builder_cls(config_name=config, hash=dataset_module.hash, **dataset_module.builder_kwargs)
        job = {
            "dataset": dataset,
            "kind": ConfigSplitNamesJobRunner.get_job_type(),
            "config": config,
            "split": None,
        }
        dataset_init_response["successes"].append(job)
        split_items = [
            FullSplitItem(dataset=dataset, config=config, split=split) for split in builder.config.data_files
        ]
        yield ShortcutCompleteJobResult(
            content=SplitsList(splits=split_items),
            job=job,
        )
        yield JobResult(dataset_init_response, progress=0.2)

        # config-parquet
        all_sizes = get_file_sizes(
            fs,
            file_paths=[
                fs._strip_protocol(data_file)
                for data_files in builder.config.data_files.values()
                for data_file in data_files
            ],
        )
        job = {
            "dataset": dataset,
            "kind": ConfigParquetJobRunner.get_job_type(),
            "config": config,
            "split": None,
        }
        first_data_file = builder.config.data_files[split_items[0]["split"]][0]
        features, first_file_num_examples, first_file_size, first_file_num_bytes = (
            retry_get_features_num_examples_size_and_num_bytes(
                first_data_file, hf_endpoint=hf_endpoint, hf_token=hf_token
            )
        )
        raise_if_long_column_name(features)
        features_dict = backward_compat_features(asdict(features))

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
        yield ShortcutCompleteJobResult(
            content=ConfigParquetResponse(parquet_files=parquet_file_items, features=features_dict, partial=False),
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
        files = [
            {
                "path": fs.resolve_path(data_file).path_in_repo,
                "size": all_sizes[fs._strip_protocol(data_file)],
                "num_rows": None,
                "metadata_path": f"{dataset}/{DATASET_SEPARATOR}/{fs.resolve_path(data_file).path_in_repo}",
            }
            for split in builder.config.data_files
            for data_file in builder.config.data_files[split]
        ]
        revision = fs.resolve_path(first_data_file).revision
        viewer = lv.Dataset(
            name=dataset,
            files=files,
            revision=revision,
            hf_token=hf_token,
            hf_endpoint=hf_endpoint,
            data_store=data_store,
            metadata_store=f"file://{parquet_metadata_directory}",
        )
        result = viewer.sync_index(max_parallelism=max_parallelism)
        # fill builder info from the parquet files
        with patch("huggingface_hub.hf_file_system.http_backoff", http_backoff_with_timeout):
            fill_builder_info(builder, hf_endpoint=hf_endpoint, hf_token=hf_token)
        # fill parquet_files_metadata correctly
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
        # prepare parquet metadata response
        job = {
            "dataset": dataset,
            "kind": ConfigParquetMetadataJobRunner.get_job_type(),
            "config": config,
            "split": None,
        }
        dataset_init_response["successes"].append(job)
        yield ShortcutCompleteJobResult(
            content=ConfigParquetMetadataResponse(
                parquet_files_metadata=parquet_files_metadata,
                features=features_dict,
                partial=False,
            ),
            job=job,
        )
        yield JobResult(dataset_init_response, progress=0.4)
        # fill info
        job = {
            "dataset": dataset,
            "kind": ConfigParquetAndInfoJobRunner.get_job_type(),
            "config": config,
            "split": None,
        }
        dataset_info = asdict(builder.info)
        if builder.info.splits:
            dataset_info["splits"] = [asdict(split_info) for split_info in builder.info.splits.values()]
        dataset_info["features"] = backward_compat_features(dataset_info["features"])
        dataset_init_response["successes"].append(job)
        yield ShortcutCompleteJobResult(
            content=ConfigParquetAndInfoResponse(
                parquet_files=parquet_file_items,
                dataset_info=dataset_info,
                estimated_dataset_info=None,
                partial=False,
            ),
            job=job,
        )
        yield JobResult(dataset_init_response, progress=0.5)

        # config-info
        config_info_response = ConfigInfoResponse(
            dataset_info=dataset_info,
            partial=False,
        )
        job = {
            "dataset": dataset,
            "kind": ConfigInfoJobRunner.get_job_type(),
            "config": config,
            "split": None,
        }
        dataset_init_response["successes"].append(job)
        yield ShortcutCompleteJobResult(
            content=config_info_response,
            job=job,
        )
        dataset_info_response = DatasetInfoResponse(
            dataset_info={config: dataset_info},
            pending=[
                CachedJob(
                    dataset=dataset,
                    config=config_name_item["config"],
                    split=None,
                    kind=ConfigSizeJobRunner.get_job_type(),
                )
                for config_name_item in config_name_items
            ],
            failed=[],
            partial=False,
        )
        job = {
            "dataset": dataset,
            "kind": DatasetInfoJobRunner.get_job_type(),
            "config": None,
            "split": None,
        }
        dataset_init_response["successes"].append(job)
        yield ShortcutJobResult(content=dataset_info_response, job=job, progress=1 / len(config_name_items))
        yield JobResult(dataset_init_response, progress=0.6)

        # config-size
        num_columns = len(dataset_info["features"])
        split_sizes = [
            SplitSize(
                dataset=dataset,
                config=config,
                split=split_info["name"],
                num_bytes_parquet_files=sum(
                    item["size"] for item in parquet_file_items if item["split"] == split_info["name"]
                ),
                num_bytes_memory=split_info.get("num_bytes", 0),
                num_rows=split_info.get("num_examples", 0),
                num_columns=num_columns,
                estimated_num_rows=None,
            )
            for split_info in dataset_info["splits"]
        ]
        config_size = ConfigSize(
            dataset=dataset,
            config=config,
            num_bytes_original_files=dataset_info.get("download_size"),
            num_bytes_parquet_files=sum(split_size["num_bytes_parquet_files"] for split_size in split_sizes),
            num_bytes_memory=sum(split_size["num_bytes_memory"] for split_size in split_sizes),
            num_rows=sum(split_size["num_rows"] for split_size in split_sizes),
            num_columns=num_columns,
            estimated_num_rows=None,
        )
        config_size_content = ConfigSizeContent(
            config=config_size,
            splits=split_sizes,
        )
        config_size_response = ConfigSizeResponse(
            size=config_size_content,
            partial=False,
        )
        job = {
            "dataset": dataset,
            "kind": ConfigSizeJobRunner.get_job_type(),
            "config": config,
            "split": None,
        }
        dataset_init_response["successes"].append(job)
        yield ShortcutCompleteJobResult(
            content=config_size_response,
            job=job,
        )
        dataset_size = DatasetSize(
            dataset=dataset,
            num_bytes_original_files=dataset_info.get("download_size"),
            num_bytes_parquet_files=sum(split_size["num_bytes_parquet_files"] for split_size in split_sizes),
            num_bytes_memory=sum(split_size["num_bytes_memory"] for split_size in split_sizes),
            num_rows=sum(split_size["num_rows"] for split_size in split_sizes),
            estimated_num_rows=None,
        )
        dataset_size_content = DatasetSizeContent(dataset=dataset_size, configs=[config_size], splits=split_sizes)
        dataset_size_response = DatasetSizeResponse(
            size=dataset_size_content,
            pending=[
                CachedJob(
                    dataset=dataset,
                    config=config_name_item["config"],
                    split=None,
                    kind=ConfigSizeJobRunner.get_job_type(),
                )
                for config_name_item in config_name_items
            ],
            failed=[],
        )
        job = {
            "dataset": dataset,
            "kind": DatasetSizeJobRunner.get_job_type(),
            "config": None,
            "split": None,
        }
        dataset_init_response["successes"].append(job)
        yield ShortcutJobResult(content=dataset_size_response, job=job, progress=1 / len(config_name_items))
        yield JobResult(dataset_init_response, progress=0.7)

        # split-is-valid (for first split only)
        split_is_valid_response = IsValidResponse(
            viewer=True,
            preview=False,
            search=len(features_dict.get("columns", [])) > 0
            if isinstance(features_dict, dict) and "columns" in features_dict
            else False,
            filter=True,
            statistics=False,
        )
        first_split = split_items[0]["split"]
        job = {
            "dataset": dataset,
            "kind": SplitIsValidJobRunner.get_job_type(),
            "config": config,
            "split": first_split,
        }
        dataset_init_response["successes"].append(job)
        yield ShortcutCompleteJobResult(
            content=split_is_valid_response,
            job=job,
        )
        job = {
            "dataset": dataset,
            "kind": ConfigIsValidJobRunner.get_job_type(),
            "config": config,
            "split": None,
        }
        dataset_init_response["successes"].append(job)
        yield ShortcutJobResult(content=split_is_valid_response, job=job, progress=1 / len(split_items))
        job = {
            "dataset": dataset,
            "kind": DatasetIsValidJobRunner.get_job_type(),
            "config": None,
            "split": None,
        }
        dataset_init_response["successes"].append(job)
        yield ShortcutJobResult(content=split_is_valid_response, job=job, progress=1 / len(config_name_items))
        yield JobResult(dataset_init_response, progress=0.8)

        # split-first-rows (for first split only)
        # TODO: smh dataset-config-names is still triggered is this step fails
        # TODO: don't require the parquet-metadata job to be stored and fetch it - instead reuse parquet_files_metadata
        # WARNING: 2026-07-31 10:28:58,587 - root - Could not compute split-first-rows shortcut for dataset='DVUser/image_statistics-17852775329062' config='default' first_split='train': Cache entry does not exist: kind='config-parquet-metadata' dataset='DVUser/image_statistics-17852775329062' config='default' split=None
        if storage_client is not None and hf_datasets_cache is not None:
            try:
                split_first_rows_response: SplitFirstRowsResponse = compute_first_rows_from_parquet_response(
                    dataset=dataset,
                    revision=revision,
                    config=config,
                    split=first_split,
                    storage_client=storage_client,
                    min_cell_bytes=first_rows_min_cell_bytes,
                    rows_max_bytes=first_rows_max_bytes,
                    rows_min_number=first_rows_min_number,
                    rows_max_number=first_rows_max_number,
                    columns_max_number=first_rows_columns_max_number,
                    hf_token=hf_token,
                    hf_endpoint=hf_endpoint,
                    max_arrow_data_in_memory=rows_index_max_arrow_data_in_memory,
                    parquet_metadata_directory=parquet_metadata_directory,
                )
                job = {
                    "dataset": dataset,
                    "kind": SplitFirstRowsJobRunner.get_job_type(),
                    "config": config,
                    "split": first_split,
                }
                dataset_init_response["successes"].append(job)
                yield ShortcutCompleteJobResult(
                    content=split_first_rows_response,
                    job=job,
                )
                yield JobResult(dataset_init_response, progress=0.9)

                split_is_valid_response["preview"] = True
                job = {
                    "dataset": dataset,
                    "kind": SplitIsValidJobRunner.get_job_type(),
                    "config": config,
                    "split": first_split,
                }
                dataset_init_response["successes"].append(job)
                yield ShortcutCompleteJobResult(
                    content=split_is_valid_response,
                    job=job,
                )
                job = {
                    "dataset": dataset,
                    "kind": ConfigIsValidJobRunner.get_job_type(),
                    "config": config,
                    "split": None,
                }
                dataset_init_response["successes"].append(job)
                yield ShortcutJobResult(content=split_is_valid_response, job=job, progress=1 / len(split_items))
                job = {
                    "dataset": dataset,
                    "kind": DatasetIsValidJobRunner.get_job_type(),
                    "config": None,
                    "split": None,
                }
                dataset_init_response["successes"].append(job)
                yield ShortcutJobResult(content=split_is_valid_response, job=job, progress=1 / len(config_name_items))
                yield JobResult(dataset_init_response, progress=1.0)
            except Exception as e:
                logging.warning(
                    f"Could not compute split-first-rows shortcut for {dataset=} {config=} {first_split=}: {e}"
                )


class DatasetInitJobRunner(DatasetJobRunnerWithDatasetsCache):
    storage_client: StorageClient

    def __init__(
        self,
        job_info: JobInfo,
        app_config: AppConfig,
        hf_datasets_cache: Path,
        storage_client: Optional[StorageClient] = None,
    ) -> None:
        super().__init__(job_info=job_info, app_config=app_config, hf_datasets_cache=hf_datasets_cache)
        self.storage_client = storage_client

    @staticmethod
    def get_job_type() -> str:
        return "dataset-init"

    def compute(self) -> Iterator[JobResult]:
        yield from compute_init_responses(
            dataset=self.dataset,
            max_num_configs=self.app_config.config_names.max_number_for_init,
            hf_endpoint=self.app_config.common.hf_endpoint,
            hf_token=self.app_config.common.hf_token,
            data_store=None,
            parquet_metadata_directory=self.app_config.parquet_metadata.storage_directory,
            max_parallelism=self.app_config.parquet_metadata.max_parallelism,
            hf_datasets_cache=self.app_config.datasets_based.hf_datasets_cache,
            storage_client=self.storage_client,
            rows_index_max_arrow_data_in_memory=self.app_config.rows_index.max_arrow_data_in_memory,
            first_rows_columns_max_number=self.app_config.first_rows.columns_max_number,
            first_rows_max_bytes=self.app_config.first_rows.max_bytes,
            first_rows_min_cell_bytes=self.app_config.first_rows.min_cell_bytes,
            first_rows_min_number=self.app_config.first_rows.min_number,
            first_rows_max_number=3,
        )
