# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from typing import Optional

from libcommon.exceptions import PreviousStepFormatError
from libcommon.simple_cache import CachedArtifactNotFoundError, get_previous_step_or_raise

from worker.dtos import (
    CompatibleLibrary,
    DatasetFormat,
    DatasetHubCacheResponse,
    DatasetLibrary,
    DatasetModality,
    JobResult,
)
from worker.job_runners.dataset.dataset_job_runner import DatasetJobRunner


def compute_hub_cache_response(dataset: str) -> tuple[DatasetHubCacheResponse, float]:
    """
    Get the content of a 'dataset-hub-cache' SSE for one specific dataset on huggingface.co.

    Its purpose is specific to the Hub, and we won't ensure backward compatibility for this step.
    It provides information about:
    - the capabilities of the dataset: preview and viewer
    - the number of rows and if the dataset is partial

    Args:
        dataset (`str`):
            A namespace (user or an organization) and a repo name separated
            by a `/`.
    Returns:
        `tuple[DatasetHubCacheResponse, float]`: The response and the progress.
    """
    logging.info(f"compute 'dataset-hub-cache' for {dataset=}")

    preview = False
    viewer = False
    progresses: list[Optional[float]] = []
    try:
        is_valid_response = get_previous_step_or_raise(kind="dataset-is-valid", dataset=dataset)
        content = is_valid_response["content"]
        if (
            "preview" not in content
            or not isinstance(content["preview"], bool)
            or "viewer" not in content
            or not isinstance(content["viewer"], bool)
        ):
            raise PreviousStepFormatError(
                "Previous step 'dataset-is-valid' did not return the expected content: 'preview', 'viewer' or 'progress'."
            )
        preview = content["preview"]
        viewer = content["viewer"]
        progresses.append(is_valid_response["progress"])
    except PreviousStepFormatError:
        raise
    except Exception:
        logging.info(f"Missing 'dataset-is-valid' response for {dataset=}. We let the fields empty.")

    partial = False
    num_rows: Optional[int] = None
    try:
        size_response = get_previous_step_or_raise(kind="dataset-size", dataset=dataset)
        content = size_response["content"]
        if (
            "partial" not in content
            or not isinstance(content["partial"], bool)
            or "size" not in content
            or "dataset" not in content["size"]
            or "num_rows" not in content["size"]["dataset"]
            or not isinstance(content["size"]["dataset"]["num_rows"], int)
            or not (
                isinstance(content["size"]["dataset"]["estimated_num_rows"], int)
                or content["size"]["dataset"]["estimated_num_rows"] is None
            )
        ):
            raise PreviousStepFormatError(
                "Previous step 'dataset-size' did not return the expected content: 'partial' or 'size.dataset.num_rows'."
            )
        partial = content["partial"]
        num_rows = content["size"]["dataset"]["estimated_num_rows"] or content["size"]["dataset"]["num_rows"]
        progresses.append(size_response["progress"])
    except PreviousStepFormatError:
        raise
    except Exception:
        logging.info(f"Missing 'dataset-size' response for {dataset=}. We let the fields empty.")

    libraries: list[DatasetLibrary] = []
    formats: list[DatasetFormat] = []
    modalities: list[DatasetModality] = []
    try:
        compatible_libraries_response = get_previous_step_or_raise(
            kind="dataset-compatible-libraries", dataset=dataset
        )
        compatible_libraries: list[CompatibleLibrary] = compatible_libraries_response["content"]["libraries"]
        libraries = [compatible_library["library"] for compatible_library in compatible_libraries]
        formats = compatible_libraries_response["content"].get("formats", [])
        progresses.append(compatible_libraries_response["progress"])
    except CachedArtifactNotFoundError:
        logging.info(f"Missing 'dataset-compatible-libraries' response for {dataset=}")
    except KeyError:
        raise PreviousStepFormatError(
            "Previous step 'dataset-compatible-libraries' did not return the expected content: 'libraries'."
        )
    except Exception:
        logging.info("Error while parsing 'dataset-compatible-libraries' response. We let the fields empty.")

    try:
        modalities_response = get_previous_step_or_raise(kind="dataset-modalities", dataset=dataset)
        modalities = modalities_response["content"]["modalities"]
        progresses.append(modalities_response["progress"])
    except CachedArtifactNotFoundError:
        logging.info(f"Missing 'dataset-modalities' response for {dataset=}")
    except KeyError:
        raise PreviousStepFormatError(
            "Previous step 'dataset-modalities' did not return the expected content: 'modalities'."
        )
    except Exception:
        logging.info("Error while parsing 'dataset-modalities' response. We let the field empty.")

    return (
        DatasetHubCacheResponse(
            preview=preview,
            viewer=viewer,
            partial=partial,
            num_rows=num_rows,
            libraries=libraries,
            formats=formats,
            modalities=modalities,
        ),
        min([0.0 if p is None else p for p in progresses], default=0.0),
    )


class DatasetHubCacheJobRunner(DatasetJobRunner):
    @staticmethod
    def get_job_type() -> str:
        return "dataset-hub-cache"

    def compute(self) -> JobResult:
        response_content, progress = compute_hub_cache_response(dataset=self.dataset)
        return JobResult(response_content, progress=progress)
