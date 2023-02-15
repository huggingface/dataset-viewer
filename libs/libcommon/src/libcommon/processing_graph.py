# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Mapping, Optional, TypedDict

InputType = Literal["dataset", "config", "split"]


class _ProcessingStepSpecification(TypedDict):
    input_type: InputType


class ProcessingStepSpecification(_ProcessingStepSpecification, total=False):
    requires: Optional[str]
    required_by_dataset_viewer: Literal[True]


@dataclass
class ProcessingStep:
    """A dataset processing step.

    It contains the details of:
    - the API endpoint
    - the cache kind (ie. the key in the cache)
    - the job type (ie. the job to run to compute the response)
    - the job parameters (mainly: ['dataset'] or ['dataset', 'config', 'split'])
    - the immediately previous step required to compute the response
    - the list of all the previous steps required to compute the response
    - the next steps (the steps which previous step is the current one)
    """

    endpoint: str
    input_type: InputType
    requires: Optional[str]
    required_by_dataset_viewer: bool
    parent: Optional[ProcessingStep]
    ancestors: List[ProcessingStep]
    children: List[ProcessingStep]

    @property
    def job_type(self) -> str:
        """The job type (ie. the job to run to compute the response)."""
        return self.endpoint

    @property
    def cache_kind(self) -> str:
        """The cache kind (ie. the key in the cache)."""
        return self.endpoint

    def get_ancestors(self) -> List[ProcessingStep]:
        """Get all the ancestors previous steps required to compute the response of the given step."""
        if len(self.ancestors) > 0:
            return self.ancestors
        if self.parent is None:
            self.ancestors = []
        else:
            parent_ancestors = self.parent.get_ancestors()
            if self in parent_ancestors:
                raise ValueError(f"Cycle detected between {self.endpoint} and {self.parent.endpoint}")
            self.ancestors = parent_ancestors + [self.parent]
        return self.ancestors


ProcessingGraphSpecification = Mapping[str, ProcessingStepSpecification]


class ProcessingGraph:
    """A graph of dataset processing steps.

    For now, the steps can have only one parent (immediate previous step), but can have multiple children
    (next steps, found automatically by traversing the graph).
    The graph can have multiple roots.

    It contains the details of:
    - the index of all the steps, identified by their endpoint
    - the first step, or roots: they don't have a previous step. This means that they will be computed first when a
      dataset is updated.
    """

    steps: Mapping[str, ProcessingStep]
    roots: List[ProcessingStep]
    required_by_dataset_viewer: List[ProcessingStep]

    def __init__(self, processing_graph_specification: ProcessingGraphSpecification):
        # TODO: validate the graph specification: endpoints must start with "/" and use only lowercase letters
        self.steps = {
            endpoint: ProcessingStep(
                endpoint=endpoint,
                input_type=specification["input_type"],
                requires=specification.get("requires"),
                required_by_dataset_viewer=specification.get("required_by_dataset_viewer", False),
                parent=None,
                ancestors=[],
                children=[],
            )
            for endpoint, specification in processing_graph_specification.items()
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

    def get_step(self, endpoint: str) -> ProcessingStep:
        """Get a step by its endpoint."""
        if endpoint not in self.steps:
            raise ValueError(f"Unknown endpoint: {endpoint}")
        return self.steps[endpoint]

    def get_step_by_job_type(self, job_type: str) -> ProcessingStep:
        # for now: the job_type is just an alias for the endpoint
        return self.get_step(job_type)

    def get_first_steps(self) -> List[ProcessingStep]:
        """Get the first steps."""
        return self.roots

    def get_steps_required_by_dataset_viewer(self) -> List[ProcessingStep]:
        """Get the steps required by the dataset viewer."""
        return self.required_by_dataset_viewer
