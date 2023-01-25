# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, TypedDict, Union

from datasets import get_dataset_config_names
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from libcommon.exceptions import CustomError

from datasets_based.workers._datasets_based_worker import DatasetsBasedWorker

ConfigsWorkerErrorCode = Literal["EmptyDatasetError", "ConfigNamesError"]


class ConfigsWorkerError(CustomError):
    """Base class for worker exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ConfigsWorkerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=str(code), cause=cause, disclose_cause=disclose_cause
        )


class EmptyDatasetError(ConfigsWorkerError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class ConfigNamesError(ConfigsWorkerError):
    """Raised when the config names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ConfigNamesError", cause, True)


class ConfigItem(TypedDict):
    dataset: str
    config: str


class ConfigsResponseContent(TypedDict):
    configs: List[ConfigItem]


def compute_configs_response(
    dataset: str,
    hf_token: Optional[str] = None,
) -> ConfigsResponseContent:
    """
    Get the response of /configs for one specific dataset on huggingface.co.
    Dataset can be private or gated if you pass an acceptable token.

    It is assumed that the dataset exists and can be accessed using the token.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
    Returns:
        `ConfigsResponseContent`: An object with the list of config names.
    <Tip>
    Raises the following errors:
        - [`~workers.configs.EmptyDatasetError`]
          The dataset is empty.
        - [`~workers.splits.ConfigNamesError`]
          If the list of configs could not be obtained using the datasets library.
    </Tip>
    """
    logging.info(f"get configs for dataset={dataset}")
    use_auth_token: Union[bool, str, None] = hf_token if hf_token is not None else False
    # get the list of splits in streaming mode
    try:
        config_items: List[ConfigItem] = [
            {"dataset": dataset, "config": str(config)}
            for config in sorted(get_dataset_config_names(path=dataset, use_auth_token=use_auth_token))
        ]
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except Exception as err:
        raise ConfigNamesError("Cannot get the config names for the dataset.", cause=err) from err
    return {"configs": config_items}


class ConfigsWorker(DatasetsBasedWorker):
    @staticmethod
    def get_job_type() -> str:
        return "/configs"

    @staticmethod
    def get_version() -> str:
        return "1.0.0"

    def compute(self) -> Mapping[str, Any]:
        return compute_configs_response(dataset=self.dataset, hf_token=self.common_config.hf_token)
