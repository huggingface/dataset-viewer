# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from pathlib import Path
from typing import Optional

import datasets.config
import datasets.data_files
import yaml
from datasets import BuilderConfig, DownloadConfig
from datasets.data_files import DataFilesDict
from datasets.load import (
    create_builder_configs_from_metadata_configs,
)
from datasets.packaged_modules import _MODULE_TO_EXTENSIONS, _PACKAGED_DATASETS_MODULES
from datasets.utils.file_utils import cached_path
from datasets.utils.metadata import MetadataConfigs
from huggingface_hub import DatasetCard, DatasetCardData, HfFileSystem, hf_hub_url, HfApi
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.hf_file_system import HfFileSystemResolvedPath
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_previous_step_or_raise
from tqdm.contrib.concurrent import thread_map

from worker.dtos import CompleteJobResult, ConfigSize, ConfigSizeResponse, SplitSize
from worker.job_runners.config.config_job_runner import ConfigJobRunner


def get_data_files_dict(
    dataset: str, config: str, module_name: str, hf_token: Optional[str] = None
) -> DataFilesDict:
    """
    Get the list of builder configs to get their (possibly simplified) data_files

    Example:

    ```python
    >>> configs = get_builder_configs("Anthropic/hh-rlhf", "json")
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
            hf_hub_url(dataset, datasets.config.REPOCARD_FILENAME, repo_type="dataset"),
            download_config=download_config,
        )
        dataset_card_data = DatasetCard.load(Path(dataset_readme_path)).data
    except FileNotFoundError:
        dataset_card_data = DatasetCardData()
    try:
        standalone_yaml_path = cached_path(
            hf_hub_url(dataset, datasets.config.REPOYAML_FILENAME, repo_type="dataset"),
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
    metadata_configs = MetadataConfigs({config: metadata_configs.get(config, {})})
    module_path, _ = _PACKAGED_DATASETS_MODULES[module_name]
    builder_configs, _ = create_builder_configs_from_metadata_configs(
        module_path,
        metadata_configs,
        supports_metadata=False,
        base_path=base_path,
        download_config=download_config,
    )
    return builder_configs[0].data_files.resolve(base_path=base_path, download_config=download_config)


def estimate_data_file_num_rows(data_file: str, module_name: str, hf_endpoint: str, hf_token: Optional[str]) -> int:
    # TODO


def estimate_split_num_rows(dataset: str, module_name: str, data_files: list[str], hf_endpoint: str, hf_token: Optional[str]) -> int:
    if module_name in ["audiofolder", "imagefolder"]:
        return len(data_files)
    else:
        paths = [data_file.split(dataset, 1)[-1].split("/", 1)[-1] for data_file in data_files]
        paths_infos =[path_info for path_info in HfApi(endpoint=hf_endpoint, token=hf_token).get_paths_info(paths=paths, repo_id=dataset, repo_type="dataset") if isinstance(path_info, RepoFile)]
        if len(paths_infos) != len(data_files):
            raise RuntimeError("Error when reading the sizes of the data_files: size of paths_infos doesn't match")
        total_size = sum(path_info.size for path_info in paths_infos)
        sample_data_files = data_files[::len(data_files) // 100]
        sample_size = sum(path_info.size for path_info in paths_infos[::len(data_files) // 100])
        return total_size / sample_size * sum(estimate_data_file_num_rows(data_file, hf_endpoint=hf_endpoint, hf_token=hf_token) for data_file in sample_data_files)


def compute_config_size_response(dataset: str, config: str, hf_endpoint: str, hf_token: Optional[str]) -> ConfigSizeResponse:
    """
    Get the response of 'config-size' for one specific dataset and config on huggingface.co.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            A configuration name.

    Raises:
        [~`libcommon.simple_cache.CachedArtifactError`]:
          If the previous step gave an error.
        [~`libcommon.exceptions.PreviousStepFormatError`]:
          If the content of the previous step has not the expected format

    Returns:
        `ConfigSizeResponse`: An object with the size_response.
    """
    logging.info(f"compute 'config-size' for {dataset=} {config=}")

    dataset_info_response = get_previous_step_or_raise(kind="config-parquet-and-info", dataset=dataset, config=config)
    content = dataset_info_response["content"]
    if "dataset_info" not in content:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'dataset_info'.")
    if not isinstance(content["dataset_info"], dict):
        raise PreviousStepFormatError(
            "Previous step did not return the expected content.",
            TypeError(f"dataset_info should be a dict, but got {type(content['dataset_info'])}"),
        )

    try:
        config_info = content["dataset_info"]
        num_columns = len(config_info["features"])
        split_sizes: list[SplitSize] = [
            {
                "dataset": dataset,
                "config": config,
                "split": split_info["name"],
                "num_bytes_parquet_files": sum(
                    x["size"]
                    for x in content["parquet_files"]
                    if x["config"] == config and x["split"] == split_info["name"]
                ),
                "num_bytes_memory": split_info["num_bytes"] if "num_bytes" in split_info else 0,
                "num_rows": split_info["num_examples"] if "num_examples" in split_info else 0,
                "num_columns": num_columns,
            }
            for split_info in config_info["splits"].values()
        ]
        config_size = ConfigSize(
            {
                "dataset": dataset,
                "config": config,
                "num_bytes_original_files": config_info.get("download_size"),
                "num_bytes_parquet_files": sum(split_size["num_bytes_parquet_files"] for split_size in split_sizes),
                "num_bytes_memory": sum(split_size["num_bytes_memory"] for split_size in split_sizes),
                "num_rows": sum(split_size["num_rows"] for split_size in split_sizes),
                "num_columns": num_columns,
            }
        )
        partial = content["partial"]
        module_name = config_info["builder_name"]
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e
    
    if partial:
        data_files_dict = get_data_files_dict(dataset=dataset, config=config, module_name=module_name)
        for split_size in split_sizes:
            data_files = [str(data_file) for data_file in data_files_dict[split_size["split"]]]
        config_size["estimated_num_rows"] = sum(split_size["estimated_num_rows"] for split_size in split_sizes)

    return ConfigSizeResponse(
        {
            "size": {
                "config": config_size,
                "splits": split_sizes,
            },
            "partial": partial,
        }
    )


class ConfigSizeJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-size"

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(compute_config_size_response(dataset=self.dataset, config=self.config, hf_endpoint=self.app_config.common.hf_endpoint, hf_token=self.app_config.common.hf_token))
