# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from http import HTTPStatus
from typing import Any, List, Literal, Mapping, Optional, TypedDict, Union

from datasets import get_dataset_config_names
from datasets.data_files import EmptyDatasetError as _EmptyDatasetError
from libcommon.constants import PROCESSING_STEP_CONFIG_NAMES_VERSION
from libcommon.simple_cache import SplitFullName

from worker.job_runner import CompleteJobResult, JobRunnerError
from worker.job_runners._datasets_based_job_runner import DatasetsBasedJobRunner

ConfigNamesJobRunnerErrorCode = Literal["EmptyDatasetError", "ConfigNamesError"]


class ConfigNamesJobRunnerError(JobRunnerError):
    """Base class for job runner exceptions."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ConfigNamesJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class EmptyDatasetError(ConfigNamesJobRunnerError):
    """Raised when the dataset has no data."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "EmptyDatasetError", cause, True)


class ConfigNamesError(ConfigNamesJobRunnerError):
    """Raised when the config names could not be fetched."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "ConfigNamesError", cause, True)


class ConfigNameItem(TypedDict):
    dataset: str
    config: str


class ConfigNamesResponseContent(TypedDict):
    config_names: List[ConfigNameItem]


def compute_config_names_response(
    dataset: str,
    hf_token: Optional[str] = None,
) -> ConfigNamesResponseContent:
    """
    Get the response of /config-names for one specific dataset on huggingface.co.
    Dataset can be private or gated if you pass an acceptable token.

    It is assumed that the dataset exists and can be accessed using the token.

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        hf_token (`str`, *optional*):
            An authentication token (See https://huggingface.co/settings/token)
    Returns:
        `ConfigNamesResponseContent`: An object with the list of config names.
    <Tip>
    Raises the following errors:
        - [`~job_runners.config_names.EmptyDatasetError`]
          The dataset is empty.
        - [`~job_runners.config_names.ConfigNamesError`]
          If the list of configs could not be obtained using the datasets library.
    </Tip>
    """
    logging.info(f"get config names for dataset={dataset}")
    use_auth_token: Union[bool, str, None] = hf_token if hf_token is not None else False
    # get the list of splits in streaming mode
    try:
        config_name_items: List[ConfigNameItem] = [
            {"dataset": dataset, "config": str(config)}
            for config in sorted(get_dataset_config_names(path=dataset, use_auth_token=use_auth_token))
        ]
    except _EmptyDatasetError as err:
        raise EmptyDatasetError("The dataset is empty.", cause=err) from err
    except Exception as err:
        raise ConfigNamesError("Cannot get the config names for the dataset.", cause=err) from err
    return {"config_names": config_name_items}


class ConfigNamesJobRunner(DatasetsBasedJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "/config-names"

    @staticmethod
    def get_job_runner_version() -> int:
        return 1

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(
            compute_config_names_response(dataset=self.dataset, hf_token=self.common_config.hf_token)
        )

    def get_new_splits(self, content: Mapping[str, Any]) -> set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {SplitFullName(dataset=s["dataset"], config=s["config"], split=None) for s in content["config_names"]}
