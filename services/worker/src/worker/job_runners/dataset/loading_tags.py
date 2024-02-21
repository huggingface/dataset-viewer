# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import re
from http import HTTPStatus
from itertools import islice
from pathlib import Path
from typing import Any, Callable, Optional

import datasets.config
import datasets.data_files
import yaml
from datasets import BuilderConfig, DownloadConfig
from datasets.data_files import (
    NON_WORDS_CHARS,
    DataFilesPatternsDict,
    DataFilesPatternsList,
    resolve_pattern,
)
from datasets.load import (
    create_builder_configs_from_metadata_configs,
)
from datasets.packaged_modules import _MODULE_TO_EXTENSIONS, _PACKAGED_DATASETS_MODULES
from datasets.utils.file_utils import cached_path
from datasets.utils.hub import hf_hub_url
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import DatasetCard, DatasetCardData, HfFileSystem
from libcommon.constants import LOADING_METHODS_MAX_CONFIGS
from libcommon.croissant_utils import get_record_set
from libcommon.exceptions import DatasetWithTooComplexDataFilesPatternsError, PreviousStepFormatError
from libcommon.simple_cache import (
    get_previous_step_or_raise,
)

from worker.dtos import (
    CompleteJobResult,
    DatasetLoadingTag,
    DatasetLoadingTagsResponse,
    DatasetPythonLibrary,
    LoadingCode,
    PythonLoadingMethod,
)
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner

NON_WORD_GLOB_SEPARATOR = f"[{NON_WORDS_CHARS}/]"
NON_WORD_REGEX_SEPARATOR = NON_WORD_GLOB_SEPARATOR.replace(".", "\.").replace("/", "\/")

if any(
    NON_WORD_GLOB_SEPARATOR not in pattern.format(keyword="train", sep=NON_WORDS_CHARS)
    for pattern in datasets.data_files.KEYWORDS_IN_PATH_NAME_BASE_PATTERNS
):
    raise ImportError(
        f"Current `datasets` version is not compatible with simplify_data_files_patterns() which expects as keyword separator {NON_WORD_GLOB_SEPARATOR} for glob patterns. "
        "Indeed the simplify_data_files_patterns() function is used to create human-readable code snippets with nice glob patterns for files, "
        f"and therefore it replaces the ugly {NON_WORD_GLOB_SEPARATOR} separator with actual characters, for example\n"
        "**/*[-._ 0-9/]train[-._ 0-9/]**    =>    **/*_train_*.jsonl\n\n"
        "To fix this error, please update the simplify_data_files_patterns() to make it support `datasets` new separator and patterns. "
        "After the fix the get_builder_configs_with_simplified_data_files() should return proper simplified data files on most datasets."
    )


def get_builder_configs_with_simplified_data_files(
    dataset: str, module_name: str, hf_token: Optional[str] = None
) -> list[BuilderConfig]:
    """
    Get the list of builder configs to get their (possibly simplified) data_files

    Example:

    ```python
    >>> configs = get_builder_configs_with_simplified_data_files("Anthropic/hh-rlhf", "json")
    >>> configs[0].data_files
    {'train': ['**/*/train.jsonl.gz'], 'test': ['**/*/test.jsonl.gz']}
    ```

    which has simpler and better looking glob patterns that what `datasets` uses by default that look like **/*[-._ 0-9/]train[-._ 0-9/]**
    """
    builder_configs: list[BuilderConfig]
    base_path = f"hf://datasets/{dataset}"
    if HfFileSystem().exists(base_path + "/" + dataset.split("/")[-1] + ".py"):
        raise NotImplementedError("datasets with a script are not supported")
    download_config = DownloadConfig(token=hf_token)
    try:
        dataset_readme_path = cached_path(
            hf_hub_url(dataset, datasets.config.REPOCARD_FILENAME),
            download_config=download_config,
        )
        dataset_card_data = DatasetCard.load(Path(dataset_readme_path)).data
    except FileNotFoundError:
        dataset_card_data = DatasetCardData()
    try:
        standalone_yaml_path = cached_path(
            hf_hub_url(dataset, datasets.config.REPOYAML_FILENAME),
            download_config=download_config,
        )
        with open(standalone_yaml_path, "r", encoding="utf-8") as f:
            standalone_yaml_data = yaml.safe_load(f.read())
            if standalone_yaml_data:
                _dataset_card_data_dict = dataset_card_data.to_dict()
                _dataset_card_data_dict.update(standalone_yaml_data)
                dataset_card_data = DatasetCardData(**_dataset_card_data_dict)
    except FileNotFoundError:
        pass
    metadata_configs = MetadataConfigs.from_dataset_card_data(dataset_card_data)
    module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
    builder_configs, _ = create_builder_configs_from_metadata_configs(
        module_path,
        metadata_configs or MetadataConfigs({"default": {}}),
        supports_metadata=False,
        base_path=base_path,
        download_config=download_config,
    )
    if not metadata_configs:  # inferred patterns are ugly, so let's simplify them
        for config in builder_configs:
            data_files = config.data_files.resolve(base_path=base_path, download_config=download_config)
            config.data_files = DataFilesPatternsDict(
                {
                    str(split): (
                        simplify_data_files_patterns(
                            data_files_patterns=config.data_files[split],
                            base_path=base_path,
                            download_config=download_config,
                            allowed_extensions=_MODULE_TO_EXTENSIONS[module_name],
                        )
                    )
                    for split in data_files
                }
            )
    return builder_configs


def simplify_data_files_patterns(
    data_files_patterns: DataFilesPatternsList,
    base_path: str,
    download_config: DownloadConfig,
    allowed_extensions: list[str],
) -> DataFilesPatternsList:
    """
    Simplify inferred data files patterns depending on the dataset repository content.
    All the possible patterns are defined in datasets.data_files.ALL_SPLIT_PATTERNS and ALL_DEFAULT_PATTERNS

    From those patterns this function tries to:

    - replace the `[0-9][0-9][0-9][0-9][0-9]` symbols by `*`
    - replace the separators `[-._ 0-9/]` by their actual character values
    - add the file extension

    For example:

        ```
        data/train-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]*.*        =>      data/train-*-of-*.parquet
        **/*[-._ 0-9/]train[-._ 0-9/]**                                             =>      **/*_train_*.jsonl
        test[-._ 0-9/]**                                                            =>      test[0-9].csv
        train[-._ 0-9/]**                                                           =>      train-*.tar
        ```
    """
    patterns = DataFilesPatternsList([], allowed_extensions=None)
    for pattern in data_files_patterns:
        if pattern == "**":
            pattern = "**/*"
        try:
            resolved_data_files = resolve_pattern(
                pattern, base_path=base_path, download_config=download_config, allowed_extensions=allowed_extensions
            )
        except FileNotFoundError:
            continue
        if len(resolved_data_files) == 1:
            # resolved paths are absolute
            return [resolved_data_files[0][len(base_path) + 1 :]]
        if resolved_data_files:
            # Do we get the same files if we replace the '[0-9]' symbols by '*' in the pattern ?
            if "[0-9]" * 5 in pattern:
                new_pattern = pattern.replace("[0-9]" * 5 + "*", "*")
                new_pattern = new_pattern.replace("[0-9]" * 5, "*")
                try:
                    re_resolved_data_files = resolve_pattern(
                        new_pattern,
                        base_path=base_path,
                        download_config=download_config,
                        allowed_extensions=allowed_extensions,
                    )
                except FileNotFoundError:
                    continue
                if len(resolved_data_files) == len(re_resolved_data_files):
                    pattern = new_pattern
            # Do we get the same files if we replace the NON_WORD_GLOB_SEPARATOR symvols by its actual character values in the pattern ?
            if NON_WORD_GLOB_SEPARATOR in pattern:
                re_match = re.match(
                    pattern.replace("**/*", ".*")
                    .replace("*", ".*")
                    .replace(NON_WORD_GLOB_SEPARATOR, f"({NON_WORD_REGEX_SEPARATOR})"),
                    resolved_data_files[0],
                )
                if re_match:
                    new_pattern = pattern
                    for non_word_char in re_match.groups():
                        if non_word_char in "1234567890":
                            non_word_char = "[0-9]"
                        new_pattern = new_pattern.replace(NON_WORD_GLOB_SEPARATOR, non_word_char, 1)
                    try:
                        re_resolved_data_files = resolve_pattern(
                            new_pattern,
                            base_path=base_path,
                            download_config=download_config,
                            allowed_extensions=allowed_extensions,
                        )
                    except FileNotFoundError:
                        continue
                    if len(resolved_data_files) == len(re_resolved_data_files):
                        pattern = new_pattern
            # Do we get the same files if we add the file extension at the end of the pattern ?
            for allowed_extension in allowed_extensions:
                new_pattern = pattern
                if new_pattern.endswith(".**"):
                    new_pattern = new_pattern[:-3] + allowed_extension
                elif new_pattern.endswith("**"):
                    new_pattern = new_pattern[:-1] + allowed_extension
                elif new_pattern.endswith(".*"):
                    new_pattern = new_pattern[:-2] + allowed_extension
                elif new_pattern.endswith("*"):
                    new_pattern = new_pattern + allowed_extension
                try:
                    re_resolved_data_files = resolve_pattern(
                        new_pattern,
                        base_path=base_path,
                        download_config=download_config,
                    )
                except FileNotFoundError:
                    # try again by adding a possible compression extension
                    new_pattern += "." + resolved_data_files[0].split(".")[-1]
                    try:
                        re_resolved_data_files = resolve_pattern(
                            new_pattern,
                            base_path=base_path,
                            download_config=download_config,
                        )
                    except FileNotFoundError:
                        continue
                if len(resolved_data_files) == len(re_resolved_data_files):
                    pattern = new_pattern
            patterns.append(pattern.replace("**.", "*."))
    return patterns


DATASETS_CODE = """from datasets import load_dataset

ds = load_dataset("{dataset}")"""

DATASETS_CODE_CONFIGS = """from datasets import load_dataset

ds = load_dataset("{dataset}", "{config_name}")"""


MLCROISSANT_CODE_RECORD_SETS = """from mlcroissant import Dataset
{comment}
ds = Dataset(jsonld="https://datasets-server.huggingface.co/croissant?dataset={dataset}")
records = ds.records("{record_set}")"""


def get_python_loading_method_for_datasets_library(dataset: str, infos: list[dict[str, Any]]) -> PythonLoadingMethod:
    return {
        "library": "datasets",
        "function": "load_dataset",
        "loading_codes": [
            {
                "config_name": info["config_name"],
                "arguments": {"config_name": info["config_name"]} if len(infos) > 1 else {},
                "code": (
                    DATASETS_CODE_CONFIGS.format(dataset=dataset, config_name=info["config_name"])
                    if len(infos) > 1
                    else DATASETS_CODE.format(dataset=dataset)
                ),
            }
            for info in infos
        ],
    }


def get_python_loading_method_for_mlcroissant_library(
    dataset: str, infos: list[dict[str, Any]], partial: bool
) -> PythonLoadingMethod:
    comment = "\n# The Croissant metadata exposes the first 5GB of this dataset" if partial else ""
    return {
        "library": "mlcroissant",
        "function": "Dataset",
        "loading_codes": [
            {
                "config_name": info["config_name"],
                "arguments": {
                    "record_set": get_record_set(dataset=dataset, config_name=info["config_name"]),
                    "partial": partial,
                },
                "code": (
                    MLCROISSANT_CODE_RECORD_SETS.format(
                        dataset=dataset,
                        record_set=get_record_set(dataset=dataset, config_name=info["config_name"]),
                        comment=comment,
                    )
                ),
            }
            for info in infos
        ],
    }


PANDAS_CODE = """import pandas as pd

df = {function}("hf://datasets/{dataset}/{data_file}"{args})"""


PANDAS_CODE_SPLITS = """import pandas as pd

splits = {splits}
df = {function}("hf://datasets/{dataset}/" + splits["{first_split}"{args}])"""


DASK_CODE = """import dask.dataframe as dd

df = {function}("hf://datasets/{dataset}/{pattern}")"""


DASK_CODE_SPLITS = """import dask.dataframe as dd

splits = {splits}
df = {function}("hf://datasets/{dataset}/" + splits["{first_split}"])"""


WEBDATASET_CODE = """import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url

fs = HfFileSystem()
files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/{dataset}/{pattern}")]
urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
urls = f"pipe: curl -s -L -H 'Authorization:Bearer {{get_token()}}' {{'::'.join(urls)}}"

ds = {function}(urls).decode()"""


WEBDATASET_CODE_SPLITS = """import webdataset as wds
from huggingface_hub import HfFileSystem, get_token, hf_hub_url

splits = {splits}

fs = HfFileSystem()
files = [fs.resolve_path(path) for path in fs.glob("hf://datasets/{dataset}/" + splits["{first_split}"])]
urls = [hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset") for file in files]
urls = f"pipe: curl -s -L -H 'Authorization:Bearer {{get_token()}}' {{'::'.join(urls)}}"

ds = {function}(urls).decode()"""


def get_python_loading_method_for_json(dataset: str, hf_token: Optional[str]) -> PythonLoadingMethod:
    library: DatasetPythonLibrary
    builder_configs = get_builder_configs_with_simplified_data_files(dataset, module_name="json", hf_token=hf_token)
    for config in builder_configs:
        if any(len(data_files) != 1 for data_files in config.data_files.values()):
            raise DatasetWithTooComplexDataFilesPatternsError(
                f"Failed to simplify json data files pattern: {config.data_files}"
            )
    loading_codes: list[LoadingCode] = [
        {
            "config_name": config.name,
            "arguments": {"splits": {str(split): data_files[0] for split, data_files in config.data_files.items()}},
            "code": "",
        }
        for config in builder_configs
    ]
    is_single_file = all(
        "*" not in data_file and "[" not in data_file
        for loading_code in loading_codes
        for data_file in loading_code["arguments"]["splits"].values()
    )
    if is_single_file:
        library = "pandas"
        function = "pd.read_json"
        for loading_code in loading_codes:
            first_file = next(iter(loading_code["arguments"]["splits"].values()))
            if ".jsonl" in first_file or HfFileSystem(token=hf_token).open(first_file, "r").read(1) != "[":
                args = ", lines=True"
                loading_code["arguments"]["lines"] = True
            else:
                args = ""
            if len(loading_code["arguments"]["splits"]) == 1:
                data_file = next(iter(loading_code["arguments"]["splits"].values()))
                loading_code["code"] = PANDAS_CODE.format(
                    function=function, dataset=dataset, data_file=data_file, args=args
                )
            else:
                loading_code["code"] = PANDAS_CODE_SPLITS.format(
                    function=function,
                    dataset=dataset,
                    splits=loading_code["arguments"]["splits"],
                    first_split=next(iter(loading_code["arguments"]["splits"])),
                    args=args,
                )
    else:
        library = "dask"
        function = "dd.read_json"
        for loading_code in loading_codes:
            if len(loading_code["arguments"]["splits"]) == 1:
                pattern = next(iter(loading_code["arguments"]["splits"].values()))
                loading_code["code"] = DASK_CODE.format(function=function, dataset=dataset, pattern=pattern)
            else:
                loading_code["code"] = DASK_CODE_SPLITS.format(
                    function=function,
                    dataset=dataset,
                    splits=loading_code["arguments"]["splits"],
                    first_split=next(iter(loading_code["arguments"]["splits"])),
                )
    return {"library": library, "function": function, "loading_codes": loading_codes}


def get_python_loading_method_for_csv(dataset: str, hf_token: Optional[str]) -> PythonLoadingMethod:
    library: DatasetPythonLibrary
    builder_configs = get_builder_configs_with_simplified_data_files(dataset, module_name="csv", hf_token=hf_token)
    for config in builder_configs:
        if any(len(data_files) != 1 for data_files in config.data_files.values()):
            raise DatasetWithTooComplexDataFilesPatternsError(
                f"Failed to simplify csv data files pattern: {config.data_files}"
            )
    loading_codes: list[LoadingCode] = [
        {
            "config_name": config.name,
            "arguments": {"splits": {str(split): data_files[0] for split, data_files in config.data_files.items()}},
            "code": "",
        }
        for config in builder_configs
    ]
    is_single_file = all(
        "*" not in data_file and "[" not in data_file
        for loading_code in loading_codes
        for data_file in loading_code["arguments"]["splits"].values()
    )
    if is_single_file:
        library = "pandas"
        function = "pd.read_csv"
        for loading_code in loading_codes:
            first_file = next(iter(loading_code["arguments"]["splits"].values()))
            if ".tsv" in first_file:
                args = ', sep="\\t"'
            else:
                args = ""
            if len(loading_code["arguments"]["splits"]) == 1:
                data_file = next(iter(loading_code["arguments"]["splits"].values()))
                loading_code["code"] = PANDAS_CODE.format(
                    function=function, dataset=dataset, data_file=data_file, args=args
                )
            else:
                loading_code["code"] = PANDAS_CODE_SPLITS.format(
                    function=function,
                    dataset=dataset,
                    splits=loading_code["arguments"]["splits"],
                    first_split=next(iter(loading_code["arguments"]["splits"])),
                    args=args,
                )
    else:
        library = "dask"
        function = "dd.read_csv"
        for loading_code in loading_codes:
            if len(loading_code["arguments"]["splits"]) == 1:
                pattern = next(iter(loading_code["arguments"]["splits"].values()))
                loading_code["code"] = DASK_CODE.format(function=function, dataset=dataset, pattern=pattern)
            else:
                loading_code["code"] = DASK_CODE_SPLITS.format(
                    function=function,
                    dataset=dataset,
                    splits=loading_code["arguments"]["splits"],
                    first_split=next(iter(loading_code["arguments"]["splits"])),
                )
    return {"library": library, "function": function, "loading_codes": loading_codes}


def get_python_loading_method_for_parquet(dataset: str, hf_token: Optional[str]) -> PythonLoadingMethod:
    library: DatasetPythonLibrary
    builder_configs = get_builder_configs_with_simplified_data_files(dataset, module_name="parquet", hf_token=hf_token)
    for config in builder_configs:
        if any(len(data_files) != 1 for data_files in config.data_files.values()):
            raise DatasetWithTooComplexDataFilesPatternsError(
                f"Failed to simplify parquet data files pattern: {config.data_files}"
            )
    loading_codes: list[LoadingCode] = [
        {
            "config_name": config.name,
            "arguments": {"splits": {str(split): data_files[0] for split, data_files in config.data_files.items()}},
            "code": "",
        }
        for config in builder_configs
    ]
    is_single_file = all(
        "*" not in data_file and "[" not in data_file
        for loading_code in loading_codes
        for data_file in loading_code["arguments"]["splits"].values()
    )
    if is_single_file:
        library = "pandas"
        function = "pd.read_parquet"
        for loading_code in loading_codes:
            if len(loading_code["arguments"]["splits"]) == 1:
                data_file = next(iter(loading_code["arguments"]["splits"].values()))
                loading_code["code"] = PANDAS_CODE.format(
                    function=function, dataset=dataset, data_file=data_file, args=""
                )
            else:
                loading_code["code"] = PANDAS_CODE_SPLITS.format(
                    function=function,
                    dataset=dataset,
                    splits=loading_code["arguments"]["splits"],
                    first_split=next(iter(loading_code["arguments"]["splits"])),
                    args="",
                )
    else:
        library = "dask"
        function = "dd.read_parquet"
        for loading_code in loading_codes:
            if len(loading_code["arguments"]["splits"]) == 1:
                pattern = next(iter(loading_code["arguments"]["splits"].values()))
                loading_code["code"] = DASK_CODE.format(function=function, dataset=dataset, pattern=pattern)
            else:
                loading_code["code"] = DASK_CODE_SPLITS.format(
                    function=function,
                    dataset=dataset,
                    splits=loading_code["arguments"]["splits"],
                    first_split=next(iter(loading_code["arguments"]["splits"])),
                )
    return {"library": library, "function": function, "loading_codes": loading_codes}


def get_python_loading_method_for_webdataset(dataset: str, hf_token: Optional[str]) -> PythonLoadingMethod:
    library: DatasetPythonLibrary
    builder_configs = get_builder_configs_with_simplified_data_files(
        dataset, module_name="webdataset", hf_token=hf_token
    )
    for config in builder_configs:
        if any(len(data_files) != 1 for data_files in config.data_files.values()):
            raise DatasetWithTooComplexDataFilesPatternsError(
                f"Failed to simplify webdataset data files pattern: {config.data_files}"
            )
    loading_codes: list[LoadingCode] = [
        {
            "config_name": config.name,
            "arguments": {"splits": {str(split): data_files[0] for split, data_files in config.data_files.items()}},
            "code": "",
        }
        for config in builder_configs
    ]
    library = "webdataset"
    function = "wds.WebDataset"
    for loading_code in loading_codes:
        if len(loading_code["arguments"]["splits"]) == 1:
            pattern = next(iter(loading_code["arguments"]["splits"].values()))
            loading_code["code"] = WEBDATASET_CODE.format(function=function, dataset=dataset, pattern=pattern)
        else:
            loading_code["code"] = WEBDATASET_CODE_SPLITS.format(
                function=function,
                dataset=dataset,
                splits=loading_code["arguments"]["splits"],
                first_split=next(iter(loading_code["arguments"]["splits"])),
            )
    return {"library": library, "function": function, "loading_codes": loading_codes}


get_python_loading_method_for_builder: dict[str, Callable[[str, Optional[str]], PythonLoadingMethod]] = {
    "webdataset": get_python_loading_method_for_webdataset,
    "json": get_python_loading_method_for_json,
    "csv": get_python_loading_method_for_csv,
    "parquet": get_python_loading_method_for_parquet,
}


loading_tag_for_library: dict[DatasetPythonLibrary, DatasetLoadingTag] = {
    "webdataset": "webdataset",
    "pandas": "pandas",
    "dask": "dask",
}


def compute_loading_tags_response(dataset: str, hf_token: Optional[str] = None) -> DatasetLoadingTagsResponse:
    """
    Get the response of 'dataset-loading-tags' for one specific dataset on huggingface.co.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated by a `/`.
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
            If the content of the previous step has not the expected format

    Returns:
        `DatasetLoadingTagsResponse`: The dataset-loading-tags response (list of tags).
    """
    logging.info(f"compute 'dataset-loading-tags' for {dataset=}")

    dataset_info_best_response = get_previous_step_or_raise(kinds=["dataset-info"], dataset=dataset)
    http_status = dataset_info_best_response.response["http_status"]
    tags: list[DatasetLoadingTag] = []
    python_loading_methods: list[PythonLoadingMethod] = []
    infos: list[dict[str, Any]] = []
    builder_name: Optional[str] = None
    if http_status == HTTPStatus.OK:
        try:
            content = dataset_info_best_response.response["content"]
            infos = list(islice(content["dataset_info"].values(), LOADING_METHODS_MAX_CONFIGS))
            partial = content["partial"]
        except KeyError as e:
            raise PreviousStepFormatError(
                "Previous step 'dataset-info' did not return the expected content.", e
            ) from e
    if infos:
        # mlcroissant library
        python_loading_methods.append(
            get_python_loading_method_for_mlcroissant_library(dataset, infos, partial=partial)
        )
        tags.append("croissant")
        # datasets library
        python_loading_methods.append(get_python_loading_method_for_datasets_library(dataset, infos))
        tags.append("hf_datasets")
        # pandas or dask or webdataset library
        builder_name = infos[0]["builder_name"]
        if builder_name in get_python_loading_method_for_builder:
            try:
                python_loading_method = get_python_loading_method_for_builder[builder_name](dataset, hf_token)
                if python_loading_method["library"] in loading_tag_for_library:
                    tags.append(loading_tag_for_library[python_loading_method["library"]])
                python_loading_methods.append(python_loading_method)
            except NotImplementedError:
                pass
    return DatasetLoadingTagsResponse(tags=tags, python_loading_methods=python_loading_methods)


class DatasetLoadingTagsJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-loading-tags"

    def compute(self) -> CompleteJobResult:
        response_content = compute_loading_tags_response(
            dataset=self.dataset, hf_token=self.app_config.common.hf_token
        )
        return CompleteJobResult(response_content)
