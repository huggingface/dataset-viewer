# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Literal, Mapping, Optional, TypedDict, Union

import networkx as nx

InputType = Literal["dataset", "config", "split"]


class _ProcessingStepSpecification(TypedDict):
    input_type: InputType


class ProcessingStepSpecification(_ProcessingStepSpecification, total=False):
    requires: Union[List[str], str, None]
    required_by_dataset_viewer: Literal[True]
    job_runner_version: int
    provides_dataset_config_names: bool
    provides_config_split_names: bool


@dataclass
class ProcessingStep:
    """A dataset processing step.

    It contains the details of:
    - the step name
    - the cache kind (ie. the key in the cache)
    - the job type (ie. the job to run to compute the response)
    - the input type ('dataset', 'config' or 'split')
    - the ancestors: all the chain of previous steps, even those that are not required, in no particular order
    - the children: steps that will be triggered at the end of the step, in no particular order.

    Beware: the children are computed from "requires", but with a subtlety: if c requires a and b, and if b requires a,
      only b will trigger c, i.e. c will be a child of a, but not of a.
    """

    name: str
    input_type: InputType
    requires: List[str]
    required_by_dataset_viewer: bool
    ancestors: List[ProcessingStep]
    children: List[ProcessingStep]
    parents: List[ProcessingStep]
    job_runner_version: int
    provides_dataset_config_names: Optional[bool] = False
    provides_config_split_names: Optional[bool] = False

    def __post_init__(self) -> None:
        if self.provides_dataset_config_names and self.input_type != "dataset":
            raise ValueError(
                f"Step {self.name} provides dataset config names but its input type is {self.input_type}."
            )
        if self.provides_config_split_names and self.input_type != "config":
            raise ValueError(f"Step {self.name} provides config split names but its input type is {self.input_type}.")

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


ProcessingGraphSpecification = Mapping[str, ProcessingStepSpecification]


def get_required_steps(requires: Union[List[str], str, None]) -> List[str]:
    if requires is None:
        return []
    return [requires] if isinstance(requires, str) else requires


class ProcessingGraph:
    """A graph of dataset processing steps.

    The steps can have multiple parents, and multiple children (next steps, found automatically by traversing the
      graph).
    The graph can have multiple roots.

    It contains the details of:
    - the index of all the steps, identified by their name
    - the first step, or roots: they don't have a previous step. This means that they will be computed first when a
      dataset is updated.
    """

    steps: Mapping[str, ProcessingStep]
    roots: List[ProcessingStep]
    required_by_dataset_viewer: List[ProcessingStep]
    topologically_ordered_steps: List[ProcessingStep]
    provide_dataset_config_names: List[ProcessingStep]
    provide_config_split_names: List[ProcessingStep]

    def __init__(self, processing_graph_specification: ProcessingGraphSpecification):
        self.steps = {
            name: ProcessingStep(
                name=name,
                input_type=specification["input_type"],
                requires=get_required_steps(specification.get("requires")),
                required_by_dataset_viewer=specification.get("required_by_dataset_viewer", False),
                provides_dataset_config_names=specification.get("provides_dataset_config_names", False),
                provides_config_split_names=specification.get("provides_config_split_names", False),
                ancestors=[],
                children=[],
                parents=[],
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
            for step_name in step.requires:
                graph.add_edge(step_name, name)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("The graph is not a directed acyclic graph.")

        for step in self.steps.values():
            step.ancestors = [self.get_step(name) for name in nx.ancestors(graph, step.name)]
        for step in self.steps.values():
            step.parents = [self.get_step(name) for name in graph.predecessors(step.name)]
            for parent in step.parents:
                parent.children.append(step)
        self.roots = [self.get_step(name) for name, degree in graph.in_degree() if degree == 0]
        self.required_by_dataset_viewer = [step for step in self.steps.values() if step.required_by_dataset_viewer]
        self.topologically_ordered_steps = [self.get_step(name) for name in nx.topological_sort(graph)]
        self.provide_dataset_config_names = [
            step for step in self.steps.values() if step.provides_dataset_config_names
        ]
        self.provide_config_split_names = [step for step in self.steps.values() if step.provides_config_split_names]

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
