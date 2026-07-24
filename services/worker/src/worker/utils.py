# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import itertools
import logging
import os
import posixpath
from collections.abc import Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import partial
from itertools import count, islice
from types import TracebackType
from typing import Any, Literal, Optional, TypeVar, Union, overload
from unittest.mock import patch
from urllib.parse import quote

import PIL
import requests
from datasets import (
    ClassLabel,
    DatasetBuilder,
    DownloadConfig,
    DownloadMode,
    Features,
    IterableDataset,
    IterableDatasetDict,
    Json,
    load_dataset,
)
from datasets.features.features import decode_nested_example
from datasets.utils.file_utils import SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL, is_relative_path
from huggingface_hub import HfFileSystem, HfFileSystemFile
from huggingface_hub.errors import RepositoryNotFoundError
from huggingface_hub.hf_api import HfApi
from libcommon.constants import CONFIG_SPLIT_NAMES_KIND, MAX_COLUMN_NAME_LENGTH
from libcommon.dtos import RowsContent
from libcommon.exceptions import (
    ConfigNotFoundError,
    DatasetNotFoundError,
    DatasetWithScriptNotSupportedError,
    PreviousStepFormatError,
    SplitNotFoundError,
    StreamingRowsError,
    TooLongColumnNameError,
)
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.utils import retry
from pyarrow import ArrowInvalid

MAX_IMAGE_PIXELS = 10_000_000_000
# ^ see https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.MAX_IMAGE_PIXELS


def _no_op_decode_example(self: Any, value: dict[str, Any], token_per_repo_id: Any = None) -> dict[str, Any]:  # noqa: ARG001
    return value


disable_video_decoding = patch("datasets.Video.decode_example", _no_op_decode_example)


def _patched_decode_nested_example(
    schema: Any, obj: Any, allow: str, token_per_repo_id: Optional[dict[str, Union[str, bool, None]]] = None
) -> Any:
    path: Optional[str] = None
    if (
        hasattr(schema, "decode_example")
        and getattr(schema, "decode", True)
        and not isinstance(schema, (ClassLabel, Json))
    ):
        if isinstance(obj, str):
            path = obj
        elif isinstance(obj, dict) and "path" in obj and isinstance(obj["path"], str):
            path = obj["path"]
        if path:
            resolved_path = resolve_hf_path(path)
            if not fnmatch(resolved_path, allow):
                raise ValueError(f"Data file doesn't belong to {allow}")
    return decode_nested_example(schema, obj, token_per_repo_id=token_per_repo_id)


class check_paths_during_decoding:
    def __init__(self, allow: str):
        self.allow = allow
        self.exit_stack = ExitStack()

    def __enter__(self) -> None:
        self.exit_stack.enter_context(
            patch(
                "datasets.features.features.decode_nested_example",
                partial(_patched_decode_nested_example, allow=self.allow),
            )
        )

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        return self.exit_stack.close()


@retry(on=[ConnectionError])
def get_rows(
    dataset: str,
    config: str,
    split: str,
    rows_max_number: int,
    token: Union[bool, str, None] = False,
    column_names: Optional[list[str]] = None,
) -> RowsContent:
    download_config = DownloadConfig(delete_extracted=True)
    PIL.Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
    ds = safe_load_dataset(
        dataset,
        name=config,
        split=split,
        streaming=True,
        token=token,
        download_config=download_config,
    )
    if column_names:
        ds = ds.select_columns(column_names)
    with disable_video_decoding:
        rows_plus_one = list(itertools.islice(safe_iter(ds, dataset=dataset), rows_max_number + 1))
    # ^^ to be able to detect if a split has exactly ROWS_MAX_NUMBER rows
    rows = rows_plus_one[:rows_max_number]
    all_fetched = len(rows_plus_one) <= rows_max_number
    if all_fetched:
        logging.debug(f"all the rows in the split have been fetched ({len(rows_plus_one)})")
    else:
        logging.debug(f"the rows in the split have been truncated ({rows_max_number} rows)")
    return RowsContent(rows=rows, all_fetched=all_fetched, truncated_columns=[])


def get_rows_or_raise(
    dataset: str,
    config: str,
    split: str,
    rows_max_number: int,
    token: Union[bool, str, None],
    column_names: Optional[list[str]] = [],
) -> RowsContent:
    try:
        return get_rows(
            dataset=dataset,
            config=config,
            split=split,
            rows_max_number=rows_max_number,
            token=token,
            column_names=column_names,
        )
    except Exception as err:
        if isinstance(err, ValueError) and "trust_remote_code" in str(err):
            raise DatasetWithScriptNotSupportedError from err
        else:
            raise StreamingRowsError(
                "Cannot load the dataset split (in streaming mode) to extract the first rows.",
                cause=err,
            ) from err


# TODO: use huggingface_hub's hf_hub_url after
# https://github.com/huggingface/huggingface_hub/issues/1082
def hf_hub_url(repo_id: str, filename: str, hf_endpoint: str, revision: str, url_template: str) -> str:
    return (hf_endpoint + url_template) % (repo_id, quote(revision, safe=""), filename)


def hffs_parquet_url(repo_id: str, config: str, split_directory: str, filename: str) -> str:
    """Construct url of a parquet file on the Hub, to be used with HfFileSystem."""
    return f"hf://datasets/{repo_id}/{config}/{split_directory}/{filename}"


def hf_hub_open_file(
    file_url: str, hf_endpoint: str, hf_token: Optional[str], revision: Optional[str] = None
) -> HfFileSystemFile:
    """Open file with the HfFileSystem."""
    fs = HfFileSystem(endpoint=hf_endpoint, token=hf_token)
    file: HfFileSystemFile = fs.open(file_url, revision=revision)
    return file


# used by `config-parquet-and-info` and `config-parquet-metadata` steps
@retry(on=[ArrowInvalid], sleeps=[0.2, 1, 1, 10, 10, 10])
def retry_on_arrow_invalid_open_file(
    file_url: str, hf_endpoint: str, hf_token: Optional[str], revision: Optional[str] = None
) -> HfFileSystemFile:
    return hf_hub_open_file(file_url=file_url, hf_endpoint=hf_endpoint, hf_token=hf_token, revision=revision)


DATASET_TYPE = "dataset"

LIST_REPO_REFS_RETRY_SLEEPS = [1, 1, 1, 10, 10]
LOCK_GIT_BRANCH_RETRY_SLEEPS = [1, 1, 1, 1, 1, 10, 10, 10, 10, 100] * 3


def create_branch(dataset: str, target_revision: str, hf_api: HfApi, committer_hf_api: HfApi) -> None:
    try:
        refs = retry(on=[requests.exceptions.ConnectionError], sleeps=LIST_REPO_REFS_RETRY_SLEEPS)(
            hf_api.list_repo_refs
        )(repo_id=dataset, repo_type=DATASET_TYPE)
        if all(ref.ref != target_revision for ref in refs.converts):
            initial_commit = hf_api.list_repo_commits(repo_id=dataset, repo_type=DATASET_TYPE)[-1].commit_id
            committer_hf_api.create_branch(
                repo_id=dataset, branch=target_revision, repo_type=DATASET_TYPE, revision=initial_commit, exist_ok=True
            )
    except RepositoryNotFoundError as err:
        raise DatasetNotFoundError("The dataset does not exist on the Hub (was deleted during job).") from err


def check_config_exists(dataset: str, config: str) -> None:
    """
    Check if dataset has a provided config. Dataset's configs are taken from 'dataset-config-names' step's cache.
    """
    config_names_response = get_previous_step_or_raise(kind="dataset-config-names", dataset=dataset)
    try:
        configs_content = config_names_response["content"]["config_names"]
    except Exception as e:
        raise PreviousStepFormatError(
            "Previous steps 'dataset-config-names' did not return the expected content.",
            e,
        ) from e

    if config not in [config_item["config"] for config_item in configs_content]:
        raise ConfigNotFoundError(f"Config '{config}' does not exist for dataset '{dataset}'")


def check_split_exists(dataset: str, config: str, split: str) -> None:
    """
    Check if dataset has a provided split in a provided config. Dataset's splits are taken from 'config-split-names'
      step's cache.
    """
    check_config_exists(dataset, config)
    split_names_response = get_previous_step_or_raise(kind="config-split-names", dataset=dataset, config=config)
    try:
        splits_content = split_names_response["content"]["splits"]
    except Exception as e:
        raise PreviousStepFormatError(
            "Previous step 'config-split-names' did not return the expected content.",
            e,
        ) from e

    if split not in [split_item["split"] for split_item in splits_content]:
        raise SplitNotFoundError(f"Split '{split}' does not exist for the config '{config}' of the dataset.")


def get_split_names(dataset: str, config: str) -> set[str]:
    split_names_response = get_previous_step_or_raise(kind=CONFIG_SPLIT_NAMES_KIND, dataset=dataset, config=config)

    split_names_content = split_names_response["content"]
    if "splits" not in split_names_content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'splits'.")

    if not isinstance(split_names_content["splits"], list):
        raise PreviousStepFormatError(
            "Previous step did not return the expected content.",
            TypeError(f"'splits' should be a list, but got {type(split_names_content['splits'])}"),
        )
    return {split_name_item["split"] for split_name_item in split_names_content["splits"]}


def raise_if_long_column_name(features: Optional[Features]) -> None:
    if features is None:
        return
    for feature_name in features:
        if len(feature_name) > MAX_COLUMN_NAME_LENGTH:
            short_name = feature_name[: MAX_COLUMN_NAME_LENGTH - 3] + "..."
            raise TooLongColumnNameError(
                f"Column name '{short_name}' is too long. It should be less than {MAX_COLUMN_NAME_LENGTH} characters."
            )


T = TypeVar("T")


@overload
def batched(it: Iterable[T], n: int) -> Iterable[list[T]]: ...


@overload
def batched(it: Iterable[T], n: int, with_indices: Literal[False]) -> Iterable[list[T]]: ...


@overload
def batched(it: Iterable[T], n: int, with_indices: Literal[True]) -> Iterable[tuple[list[int], list[T]]]: ...


def batched(
    it: Iterable[T], n: int, with_indices: bool = False
) -> Union[Iterable[list[T]], Iterable[tuple[list[int], list[T]]]]:
    it, indices = iter(it), count()
    while batch := list(islice(it, n)):
        yield (list(islice(indices, len(batch))), batch) if with_indices else batch


FileExtensionTuple = tuple[str, Optional[str]]


@dataclass
class FileExtension:
    extension: str
    uncompressed_extension: Optional[str] = field(default=None)

    def get_tuples(self) -> list[FileExtensionTuple]:
        """
        Get the extension and the archived extension if it exists.

        The list contains two entries if the uncompressed extension exists (for the compressed and the uncompressed files),
          otherwise one entry.
        """
        if self.uncompressed_extension:
            return [
                (self.extension, None),
                (self.uncompressed_extension, self.extension),
            ]
        return [(self.extension, None)]


def get_file_extension(filename: str, recursive: bool = True, clean: bool = True) -> FileExtension:
    """
    Get the extension of a file.

    In the case of .tar.gz or other "double extensions", the uncompressed file extension is set in the uncompressed_extension field

    Args:
        filename (`str`): The name of the file.
        recursive (`bool`, *optional*): Whether to recursively extract the extension of the archive.
        clean (`bool`, *optional*): Whether to clean the extension by removing special characters.

    Returns:
        FileExtension: the extension of the file
    """
    [base, extension] = os.path.splitext(filename)
    extension = extension.lower()
    # special cases we find in datasets (gz?dl=1 -> gz, txt_1 -> txt, txt-00000-of-00100-> txt)
    # https://github.com/huggingface/datasets/blob/af3acfdfcf76bb980dbac871540e30c2cade0cf9/src/datasets/utils/file_utils.py#L795
    if clean:
        for symb in "?-_":
            extension = extension.split(symb)[0]
    if recursive and extension.lstrip(".") in SINGLE_FILE_COMPRESSION_EXTENSION_TO_PROTOCOL:
        uncompressed_extension = get_file_extension(base, recursive=False, clean=False)
        return FileExtension(extension=extension, uncompressed_extension=uncompressed_extension.extension)
    return FileExtension(extension=extension)


def safe_load_dataset_builder(
    path: str,
    name: str,
    revision: Optional[str] = None,
    download_config: Optional[DownloadConfig] = None,
    token: Optional[Union[bool, str]] = None,
    **kwargs: Any,
) -> DatasetBuilder:
    """Simplified load_dataset_builder from `datasets` and with safety checks on data_files

    Returns:
        [`DatasetBuilder`]

    Example:

    ```py
    >>> from datasets import load_dataset_builder
    >>> ds_builder = load_dataset_builder('cornell-movie-review-data/rotten_tomatoes')
    >>> ds_builder.info.features
    {'label': ClassLabel(names=['neg', 'pos']),
     'text': Value('string')}
    ```
    """
    from datasets.load import dataset_module_factory, get_dataset_builder_class

    if path.count("/") != 1 or ".." in path or path.startswith("/"):
        raise ValueError(f"Invalid dataset: {path}")
    for key in kwargs:
        if key == "download_mode":
            if kwargs[key] is not None and kwargs[key] != DownloadMode.REUSE_DATASET_IF_EXISTS:
                raise ValueError(f"not supported in safe_load_dataset_builder: {key}")
        elif kwargs[key] is not None:
            raise ValueError(f"not supported in safe_load_dataset_builder: {key}")
    if token is not None:
        download_config = download_config.copy() if download_config else DownloadConfig()
        download_config.token = token
    repo_dir = f"hf://datasets/{path}"

    dataset_module = dataset_module_factory(
        repo_dir,
        revision=revision,
        download_config=download_config,
    )
    # Get dataset builder class
    repo_dir_with_commit_hash = repo_dir + f"@{dataset_module.hash}"
    builder_kwargs = dataset_module.builder_kwargs
    config_name = builder_kwargs.pop(
        "config_name", name or dataset_module.builder_configs_parameters.default_config_name
    )
    dataset_name = builder_kwargs.pop("dataset_name", None)
    info = dataset_module.dataset_infos.get(config_name) if dataset_module.dataset_infos else None

    builder_cls = get_dataset_builder_class(dataset_module, dataset_name=dataset_name)

    # Safety checks
    config_data_files = builder_cls.builder_configs[name].data_files
    if config_data_files is not None:
        for split in config_data_files:
            for data_file in config_data_files[split]:
                resolved_data_file = resolve_hf_path(
                    posixpath.join(builder_kwargs["base_path"], data_file)
                    if is_relative_path(data_file)
                    else data_file
                )
                if not resolved_data_file.startswith(repo_dir_with_commit_hash + "/"):
                    raise ValueError(f"Data files don't belong to {repo_dir}")

    # Instantiate the dataset builder
    builder_instance: DatasetBuilder = builder_cls(
        dataset_name=dataset_name,
        config_name=config_name,
        hash=dataset_module.hash,
        info=info,
        token=token,
        **builder_kwargs,
    )

    return builder_instance


# `datasets.inspect` binds `load_dataset_builder` at import time, so `datasets.load` is not the
# right patch target for `get_dataset_config_info` and `get_dataset_split_names`
safe_inspect = patch("datasets.inspect.load_dataset_builder", safe_load_dataset_builder)


@overload
def safe_load_dataset(
    path: str,
    name: str,
    streaming: Literal[True],
    split: None = None,
    revision: Optional[str] = None,
    download_config: Optional[DownloadConfig] = None,
    token: Optional[Union[bool, str]] = None,
) -> IterableDatasetDict: ...


@overload
def safe_load_dataset(
    path: str,
    name: str,
    streaming: Literal[True],
    split: str,
    revision: Optional[str] = None,
    download_config: Optional[DownloadConfig] = None,
    token: Optional[Union[bool, str]] = None,
) -> IterableDataset: ...


def safe_load_dataset(
    path: str,
    name: str,
    streaming: Literal[True],
    split: Optional[str] = None,
    revision: Optional[str] = None,
    download_config: Optional[DownloadConfig] = None,
    token: Optional[Union[bool, str]] = None,
) -> Union[IterableDatasetDict, IterableDataset]:
    # disallow non-streaming to not have to allow reading local files for downloaded files
    if not streaming:
        raise ValueError("Safely loading a dataset is only available in streaming mode")
    with patch("datasets.load.load_dataset_builder", safe_load_dataset_builder):
        return load_dataset(
            path,
            name=name,
            revision=revision,
            split=split,
            streaming=streaming,
            download_config=download_config,
            token=token,
        )


def safe_iter(ds: IterableDataset, dataset: str) -> Iterator[dict[str, Any]]:
    with check_paths_during_decoding(allow=f"hf://datasets/{dataset}@*/**"):
        yield from ds.decode(False) if ds.features else ds


def resolve_hf_path(path: str) -> str:
    if not path.startswith("hf://"):
        raise ValueError(f"not an hf path: {path}")
    return "hf://" + posixpath.relpath(path, start="hf://")
