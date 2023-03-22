import logging
from http import HTTPStatus
from typing import Any, Dict, Literal, Mapping, Optional, Set, TypedDict

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


# class ConfigInfo(TypedDict):
#     dataset: str
#     config: str


class ConfigInfoResponse(TypedDict):
    dataset_info: Dict[str, Any]  # info? dataset_info?


def compute_config_info_response(dataset: str, config: str) -> ConfigInfoResponse:
    logging.info(f"get dataset_info for {dataset=} and {config=}")

    try:
        response = get_response(kind="/parquet-and-dataset-info", dataset=dataset)
    except DoesNotExist as e:
        raise DatasetNotFoundError(
            "No response found in previous step for this dataset: '/parquet-and-dataset-info'.", e
        ) from e

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
        return 1

    def compute(self) -> CompleteJobResult:
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
