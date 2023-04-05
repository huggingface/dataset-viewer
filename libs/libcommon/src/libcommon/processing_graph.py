# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Literal, Mapping, Optional, TypedDict

import networkx as nx

InputType = Literal["dataset", "config", "split"]


class _ProcessingStepSpecification(TypedDict):
    input_type: InputType


class ProcessingStepSpecification(_ProcessingStepSpecification, total=False):
    requires: Optional[str]
    required_by_dataset_viewer: Literal[True]
    job_runner_version: int


@dataclass
class ProcessingStep:
    """A dataset processing step.

    It contains the details of:
    - the step name
    - the cache kind (ie. the key in the cache)
    - the job type (ie. the job to run to compute the response)
    - the job parameters (mainly: ['dataset'] or ['dataset', 'config', 'split'])
    - the immediately previous step required to compute the response
    - the list of all the previous steps required to compute the response (in no particular order)
    - the next steps (the steps which previous step is the current one, in no particular order)
    """

    name: str
    input_type: InputType
    requires: Optional[str]
    required_by_dataset_viewer: bool
    parent: Optional[ProcessingStep]
    ancestors: List[ProcessingStep]
    children: List[ProcessingStep]
    job_runner_version: int

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
                requires=specification.get("requires"),
                required_by_dataset_viewer=specification.get("required_by_dataset_viewer", False),
                parent=None,
                ancestors=[],
                children=[],
                job_runner_version=specification["job_runner_version"],
            )
            for name, specification in processing_graph_specification.items()
        }
        self.setup()

    def setup(self) -> None:
        """Setup the graph."""
        graph = nx.DiGraph()
        for name, step in self.steps.items():
            graph.add_node(name)
            if step.requires:
                graph.add_edge(step.requires, name)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("The graph is not a directed acyclic graph.")

        for step in self.steps.values():
            if parents := set(graph.predecessors(step.name)):
                if len(parents) > 1:
                    raise ValueError(f"Step {step.name} has multiple parents: {parents}")
                step.parent = self.get_step(parents.pop())
            step.children = [self.get_step(name) for name in graph.successors(step.name)]
            step.ancestors = [self.get_step(name) for name in nx.ancestors(graph, step.name)]
        self.roots = [self.get_step(name) for name, degree in graph.in_degree() if degree == 0]
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
