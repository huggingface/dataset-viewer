# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class Parameters(Enum):
    DATASET = auto()
    SPLIT = auto()


@dataclass
class ProcessingStep:
    """A dataset processing step.

    It contains the details of:
    - the API endpoint
    - the cache kind (ie. the key in the cache)
    - the job type (ie. the job to run to compute the response)
    - the job parameters (mainly: ['dataset'] or ['dataset', 'config', 'split'])
    - the previous step required to compute the response
    - the next steps (the steps which previous step is the current one)
    """

    endpoint: str
    parameters: Parameters
    previous_step: Optional[ProcessingStep]
    next_steps: List[ProcessingStep]

    @property
    def job_type(self):
        """The job type (ie. the job to run to compute the response)."""
        return self.endpoint

    @property
    def cache_kind(self):
        """The cache kind (ie. the key in the cache)."""
        return self.endpoint


SPLITS_STEP = ProcessingStep(endpoint="/splits", parameters=Parameters.DATASET, previous_step=None, next_steps=[])
PARQUET_STEP = ProcessingStep(endpoint="/parquet", parameters=Parameters.DATASET, previous_step=None, next_steps=[])
FIRST_ROWS_STEP = ProcessingStep(
    endpoint="/first-rows",
    parameters=Parameters.SPLIT,
    previous_step=SPLITS_STEP,
    next_steps=[],
)
PROCESSING_STEPS: List[ProcessingStep] = [
    SPLITS_STEP,
    PARQUET_STEP,
    FIRST_ROWS_STEP,
]
# fill the next_steps attribute
for step in PROCESSING_STEPS:
    if step.previous_step is not None:
        step.previous_step.next_steps.append(step)

# /valid and /is-valid indicate whether the dataset viewer will work
PROCESSING_STEPS_FOR_VALID: List[ProcessingStep] = [
    SPLITS_STEP,
    FIRST_ROWS_STEP,
]

INIT_PROCESSING_STEPS = [step for step in PROCESSING_STEPS if step.previous_step is None]
