# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from dataclasses import dataclass, field
from typing import List, Set, TypedDict

from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import get_valid_datasets
from starlette.requests import Request
from starlette.responses import Response

from api.utils import (
    Endpoint,
    UnexpectedError,
    get_json_api_error_response,
    get_json_ok_response,
)


class ValidContent(TypedDict):
    valid: List[str]
    preview: List[str]
    viewer: List[str]


@dataclass
class ValidDatasets:
    processing_graph: ProcessingGraph
    content: ValidContent = field(init=False)

    def __post_init__(self) -> None:
        _viewer_set: Set[str] = self._get_valid_set(
            processing_steps=self.processing_graph.get_processing_steps_enables_viewer()
        )
        _preview_set: Set[str] = self._get_valid_set(
            processing_steps=self.processing_graph.get_processing_steps_enables_preview()
        ).difference(_viewer_set)
        _valid_set = set.union(_viewer_set, _preview_set)
        self.content = ValidContent(
            valid=sorted(_valid_set),
            preview=sorted(_preview_set),
            viewer=sorted(_viewer_set),
        )

    def _get_valid_set(self, processing_steps: List[ProcessingStep]) -> Set[str]:
        """Returns the list of the valid datasets for the list of steps

        A dataset is considered valid if at least one response of any of the artifacts for any of the
        steps is valid.

        Args:
            processing_steps (List[ProcessingStep]): The list of processing steps

        Returns:
            List[str]: The list of valid datasets for the steps
        """
        if not processing_steps:
            return set()
        return set.union(
            *[get_valid_datasets(kind=processing_step.cache_kind) for processing_step in processing_steps]
        )


def create_valid_endpoint(
    processing_graph: ProcessingGraph,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    # this endpoint is used by the frontend to know which datasets support the dataset viewer
    async def valid_endpoint(_: Request) -> Response:
        with StepProfiler(method="valid_endpoint", step="all"):
            try:
                logging.info("/valid")
                with StepProfiler(method="valid_endpoint", step="prepare content"):
                    content = ValidDatasets(processing_graph=processing_graph).content
                with StepProfiler(method="valid_endpoint", step="generate OK response"):
                    return get_json_ok_response(content, max_age=max_age_long)
            except Exception as e:
                with StepProfiler(method="valid_endpoint", step="generate API error response"):
                    return get_json_api_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age_short)

    return valid_endpoint
