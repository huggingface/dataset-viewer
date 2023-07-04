import logging

from libcommon.constants import PROCESSING_STEP_CONFIG_INFO_VERSION
from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import get_previous_step_or_raise

from worker.dtos import CompleteJobResult, ConfigInfoResponse
from worker.job_runners.config.config_job_runner import ConfigJobRunner


def compute_config_info_response(dataset: str, config: str) -> ConfigInfoResponse:
    """
    Get the response of config-info for one specific config of a specific dataset on huggingface.co.
    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
        config (`str`):
            Dataset configuration name
    Returns:
        `ConfigInfoResponse`: An object with the dataset_info response for requested config.
    Raises the following errors:
        - [`libcommon.simple_cache.CachedArtifactError`]
          If the previous step gave an error.
        - [`libcommon.exceptions.PreviousStepFormatError`]
          If the content of the previous step doesn't have the expected format.
    """
    logging.info(f"get dataset_info for {dataset=} and {config=}")
    previous_step = "config-parquet-and-info"
    dataset_info_best_response = get_previous_step_or_raise(kinds=[previous_step], dataset=dataset, config=config)
    content = dataset_info_best_response.response["content"]
    try:
        config_info = content["dataset_info"]
        partial = content["partial"]
    except Exception as e:
        raise PreviousStepFormatError(
            f"Previous step '{previous_step}' did not return the expected content: 'dataset_info'.", e
        ) from e

    if not isinstance(config_info, dict):
        raise PreviousStepFormatError(
            "Previous step did not return the expected content.",
            TypeError(f"dataset_info should be a dict, but got {type(config_info)}"),
        )

    return ConfigInfoResponse(dataset_info=config_info, partial=partial)


class ConfigInfoJobRunner(ConfigJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "config-info"

    @staticmethod
    def get_job_runner_version() -> int:
        return PROCESSING_STEP_CONFIG_INFO_VERSION

    def compute(self) -> CompleteJobResult:
        return CompleteJobResult(compute_config_info_response(dataset=self.dataset, config=self.config))
