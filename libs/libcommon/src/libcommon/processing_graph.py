# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Literal, Mapping, Optional, TypedDict

InputType = Literal["dataset", "config", "split"]
AggregationLevel = Literal["dataset", "config", "split"]


class _ProcessingStepSpecification(TypedDict):
    input_type: InputType


class ProcessingStepSpecification(_ProcessingStepSpecification, total=False):
    requires: Optional[str]
    required_by_dataset_viewer: Literal[True]
    aggregation_level: Optional[AggregationLevel]


@dataclass
class ProcessingStep:
    """A dataset processing step.

    It contains the details of:
    - the step name
    - the cache kind (ie. the key in the cache)
    - the job type (ie. the job to run to compute the response)
    - the job parameters (mainly: ['dataset'] or ['dataset', 'config', 'split'])
    - the immediately previous step required to compute the response
    - the list of all the previous steps required to compute the response
    - the next steps (the steps which previous step is the current one)
    """

    name: str
    input_type: InputType
    requires: Optional[str]
    required_by_dataset_viewer: bool
    parent: Optional[ProcessingStep]
    ancestors: List[ProcessingStep]
    children: List[ProcessingStep]
    aggregation_level: Optional[AggregationLevel]

    @property
    def endpoint(self) -> str:
        warnings.warn("The use of endpoint is deprecated, name will be used instead.", category=DeprecationWarning)
        return self.name

    @property
    def job_type(self) -> str:
        """The job type (ie. the job to run to compute the response)."""
        return self.name

    @property
    def cache_kind(self) -> str:
        """The cache kind (ie. the key in the cache)."""
        return self.name

    def get_ancestors(self) -> List[ProcessingStep]:
        """Get all the ancestors previous steps required to compute the response of the given step."""
        if len(self.ancestors) > 0:
            return self.ancestors
        if self.parent is None:
            self.ancestors = []
        else:
            parent_ancestors = self.parent.get_ancestors()
            if self in parent_ancestors:
                raise ValueError(f"Cycle detected between {self.job_type} and {self.parent.job_type}")
            self.ancestors = parent_ancestors + [self.parent]
        return self.ancestors


ProcessingGraphSpecification = Mapping[str, ProcessingStepSpecification]


class ProcessingGraph:
    """A graph of dataset processing steps.

    For now, the steps can have only one parent (immediate previous step), but can have multiple children
    (next steps, found automatically by traversing the graph).
    The graph can have multiple roots.

    It contains the details of:
    - the index of all the steps, identified by their name
    - the first step, or roots: they don't have a previous step. This means that they will be computed first when a
      dataset is updated.
    """

    steps: Mapping[str, ProcessingStep]
    roots: List[ProcessingStep]
    required_by_dataset_viewer: List[ProcessingStep]

    def __init__(self, processing_graph_specification: ProcessingGraphSpecification):
        self.steps = {
            name: ProcessingStep(
                name=name,
                input_type=specification["input_type"],
                aggregation_level=specification.get("aggregation_level"),
                requires=specification.get("requires"),
                required_by_dataset_viewer=specification.get("required_by_dataset_viewer", False),
                parent=None,
                ancestors=[],
                children=[],
            )
            for name, specification in processing_graph_specification.items()
        }
        self.setup()

    def setup(self) -> None:
        """Setup the graph."""
        for step in self.steps.values():
            # Set the parent and the children
            if step.requires:
                step.parent = self.get_step(step.requires)
                step.parent.children.append(step)
            # Set the ancestors (allows to check for cycles)
            step.get_ancestors()
        self.roots = [step for step in self.steps.values() if step.parent is None]
        self.required_by_dataset_viewer = [step for step in self.steps.values() if step.required_by_dataset_viewer]

    def get_step(self, name: str) -> ProcessingStep:
        """Get a step by its name."""
        if name not in self.steps:
            raise ValueError(f"Unknown name: {name}")
        return self.steps[name]

    def get_step_by_job_type(self, job_type: str) -> ProcessingStep:
        # for now: the job_type is just an alias for the step name
        return self.get_step(job_type)

    def get_first_steps(self) -> List[ProcessingStep]:
        """Get the first steps."""
        return self.roots

    def get_steps_required_by_dataset_viewer(self) -> List[ProcessingStep]:
        """Get the steps required by the dataset viewer."""
        return self.required_by_dataset_viewer
