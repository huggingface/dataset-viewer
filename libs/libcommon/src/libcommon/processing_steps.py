# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List


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
    - the other steps required to compute the response (ie. the dependencies)
    """

    endpoint: str
    job_type: str
    cache_kind: str
    parameters: Parameters
    dependencies: List[ProcessingStep]


splits_step = ProcessingStep(
    endpoint="/splits", job_type="/splits", cache_kind="/splits", parameters=Parameters.DATASET, dependencies=[]
)
parquet_step = ProcessingStep(
    endpoint="/parquet", job_type="/parquet", cache_kind="/parquet", parameters=Parameters.DATASET, dependencies=[]
)
first_rows_step = ProcessingStep(
    endpoint="/first-rows",
    job_type="/first-rows",
    cache_kind="/first-rows",
    parameters=Parameters.SPLIT,
    dependencies=[splits_step],
)

PROCESSING_STEPS: List[ProcessingStep] = [
    splits_step,
    parquet_step,
    first_rows_step,
]

PROCESSING_STEPS_FOR_VALID: List[ProcessingStep] = [
    splits_step,
    first_rows_step,
]

INIT_PROCESSING_STEPS = [step for step in PROCESSING_STEPS if step.parameters == Parameters.DATASET]
