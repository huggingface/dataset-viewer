import logging
from http import HTTPStatus
from typing import Any, Dict, Literal, Mapping, Optional, Set, TypedDict

from libcommon.constants import PROCESSING_STEP_CONFIG_INFO_VERSION
from libcommon.dataset import DatasetNotFoundError
from libcommon.simple_cache import DoesNotExist, SplitFullName, get_response

from worker.job_runner import CompleteJobResult, JobRunner, JobRunnerError
from worker.job_runners.parquet_and_dataset_info import ParquetAndDatasetInfoResponse

ConfigInfoJobRunnerErrorCode = Literal[
    "PreviousStepStatusError",
    "PreviousStepFormatError",
    "MissingInfoForConfigError",
]


class ConfigInfoJobRunnerError(JobRunnerError):
    """Base class for exceptions in this module."""

    def __init__(
        self,
        message: str,
        status_code: HTTPStatus,
        code: ConfigInfoJobRunnerErrorCode,
        cause: Optional[BaseException] = None,
        disclose_cause: bool = False,
    ):
        super().__init__(
            message=message, status_code=status_code, code=code, cause=cause, disclose_cause=disclose_cause
        )


class PreviousStepStatusError(ConfigInfoJobRunnerError):
    """Raised when the previous step gave an error. The job should not have been created."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepStatusError", cause, False)


class PreviousStepFormatError(ConfigInfoJobRunnerError):
    """Raised when the content of the previous step has not the expected format."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "PreviousStepFormatError", cause, False)


class MissingInfoForConfigError(ConfigInfoJobRunnerError):
    """Raised when the dataset info from the parquet export is missing the requested dataset configuration."""

    def __init__(self, message: str, cause: Optional[BaseException] = None):
        super().__init__(message, HTTPStatus.INTERNAL_SERVER_ERROR, "MissingInfoForConfigError", cause, False)


class ConfigInfoResponse(TypedDict):
    dataset_info: Dict[str, Any]


def compute_config_info_response(dataset: str, config: str) -> ConfigInfoResponse:
    """
    Get the response of config-info for one specific config of a specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        `ConfigInfoResponse`: An object with the dataset_info response for requested config.
    <Tip>
    Raises the following errors:
        - [`~job_runners.config_info.PreviousStepStatusError`]
        `If the previous step gave an error.
        - [`~job_runners.config_info.PreviousStepFormatError`]
            If the content of the previous step doesn't have the expected format
        - [`~job_runners.config_info.MissingInfoForConfigError`]
            If the dataset info from the parquet export doesn't have the requested dataset configuration
        - [`~libcommon.dataset.DatasetNotFoundError`]
            If the dataset does not exist, or if the
            token does not give the sufficient access to the dataset, or if the dataset is private
            (private datasets are not supported by the datasets server)
    </Tip>
    """
    logging.info(f"get dataset_info for {dataset=} and {config=}")

    try:
        response = get_response(kind="/parquet-and-dataset-info", dataset=dataset)
    except DoesNotExist as e:
        raise DatasetNotFoundError(
            "No response found in previous step for this dataset: '/parquet-and-dataset-info'.", e
        ) from e

    if response["http_status"] != HTTPStatus.OK:
        raise PreviousStepStatusError(f"Previous step raised an error: {response['http_status']}..")

    try:
        content = ParquetAndDatasetInfoResponse(
            parquet_files=response["content"]["parquet_files"], dataset_info=response["content"]["dataset_info"]
        )
    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content: 'dataset_info'.", e) from e

    if config not in content["dataset_info"]:
        if not isinstance(content["dataset_info"], dict):
            raise PreviousStepFormatError(
                "Previous step did not return the expected content.",
                TypeError(f"dataset_info should be a dict, but got {type(content['dataset_info'])}"),
            )
        raise MissingInfoForConfigError(
            f"Dataset configuration '{config}' is missing in the dataset info from the parquet export. "
            f"Available configurations: {', '.join(list(content['dataset_info'])[:10])}"
            + f"... ({len(content['dataset_info']) - 10})"
            if len(content["dataset_info"]) > 10
            else ""
        )
    try:
        config_info = content["dataset_info"][config]

    except Exception as e:
        raise PreviousStepFormatError("Previous step did not return the expected content.", e) from e

    return ConfigInfoResponse(dataset_info=config_info)


class ConfigInfoJobRunner(JobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-info"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_INFO_VERSION

    def compute(self) -> CompleteJobResult:
        if self.dataset is None:
            raise ValueError("dataset is required")
        if self.config is None:
            raise ValueError("config is required")
        return CompleteJobResult(compute_config_info_response(dataset=self.dataset, config=self.config))

    # TODO: is it needed?
    def get_new_splits(self, content: Mapping[str, Any]) -> Set[SplitFullName]:
        """Get the set of new splits, from the content created by the compute."""
        return {
            SplitFullName(dataset=self.dataset, config=self.config, split=split)
            for split in content["dataset_info"]["splits"]
        }
