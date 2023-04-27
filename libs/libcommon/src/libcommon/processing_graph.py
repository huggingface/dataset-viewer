# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Mapping, Optional, TypedDict, Union

import networkx as nx

InputType = Literal["dataset", "config", "split"]


class _ProcessingStepSpecification(TypedDict):
    input_type: InputType


class ProcessingStepSpecification(_ProcessingStepSpecification, total=False):
    triggered_by: Union[List[str], str, None]
    required_by_dataset_viewer: Literal[True]
    job_runner_version: int
    provides_dataset_config_names: bool
    provides_config_split_names: bool


@dataclass
class ProcessingStep:
    """A dataset processing step.

    Attributes:
        name (str): The step name.
        input_type (InputType): The input type ('dataset', 'config' or 'split').
        job_runner_version (int): The version of the job runner to use to compute the response.
        triggered_by (List[str], optional): The names of the steps that trigger this step, in no particular order.
          If None, the step is a root. Defaults to None.
        parents (List[str], optional): The names of the steps that trigger this step, in no particular order.
          Defaults to [].
        ancestors (List[str], optional): All the chain of previous steps names, even those that do not trigger the
          step directly, in no particular order. Defaults to [].
        children (List[str], optional): Names of the steps that will be triggered at the end of the step, in no
          particular order. Defaults to [].
        required_by_dataset_viewer (bool, optional): Whether the step is required by the dataset viewer. Defaults to
          False.
        provides_dataset_config_names (bool, optional): Whether the step provides dataset config names. Defaults to
          False.
        provides_config_split_names (bool, optional): Whether the step provides config split names. Defaults to False.

    Getters:
        job_type (str): The job type (ie. the job to run to compute the response).
        cache_kind (str): The cache kind (ie. the key in the cache).

    Raises:
        ValueError: If the step provides dataset config names but its input type is not 'dataset', or if the step
          provides config split names but its input type is not 'config'.
    """

    name: str
    input_type: InputType
    job_runner_version: int
    triggered_by: List[str] = field(default_factory=list)
    parents: List[str] = field(default_factory=list)
    ancestors: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    required_by_dataset_viewer: bool = False
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
    def job_type(self) -> str:
        """The job type (ie. the job to run to compute the response)."""
        return self.name

    @property
    def cache_kind(self) -> str:
        """The cache kind (ie. the key in the cache)."""
        return self.name


ProcessingGraphSpecification = Mapping[str, ProcessingStepSpecification]


def get_triggered_by_as_list(triggered_by: Union[List[str], str, None]) -> List[str]:
    if triggered_by is None:
        return []
    return [triggered_by] if isinstance(triggered_by, str) else triggered_by


class ProcessingGraph:
    """A graph of dataset processing steps.

    The steps can have multiple parents, and multiple children (next steps, found automatically by traversing the
      graph).
    The graph can have multiple roots.

    Args:
        processing_graph_specification (ProcessingGraphSpecification): The specification of the graph.

    Attributes:
        steps (Mapping[str, ProcessingStep]): The steps of the graph, identified by their name.
        roots (List[str]): The names of the first steps of the graph, or roots: they don't have a previous step. This
            means that they will be computed first when a dataset is updated.
        required_by_dataset_viewer (List[str]): The names of the steps that are required by the dataset viewer.
        topologically_ordered_steps (List[str]): The names of the steps, ordered topologically.
        provide_dataset_config_names (List[str]): The names of the steps that provide dataset config names.
        provide_config_split_names (List[str]): The names of the steps that provide config split names.

    Raises:
        ValueError: If the graph is not a DAG.
    """

    steps: Mapping[str, ProcessingStep]
    roots: List[str]
    required_by_dataset_viewer: List[str]
    topologically_ordered_steps: List[str]
    provide_dataset_config_names: List[str]
    provide_config_split_names: List[str]

    def __init__(self, processing_graph_specification: ProcessingGraphSpecification):
        self.steps = {
            name: ProcessingStep(
                name=name,
                input_type=specification["input_type"],
                triggered_by=get_triggered_by_as_list(specification.get("triggered_by")),
                required_by_dataset_viewer=specification.get("required_by_dataset_viewer", False),
                provides_dataset_config_names=specification.get("provides_dataset_config_names", False),
                provides_config_split_names=specification.get("provides_config_split_names", False),
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
            for step_name in step.triggered_by:
                graph.add_edge(step_name, name)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("The graph is not a directed acyclic graph.")

        for step in self.steps.values():
            step.ancestors = list(nx.ancestors(graph, step.name))
        for step in self.steps.values():
            step.parents = list(graph.predecessors(step.name))
            for parent in step.parents:
                self.get_step(parent).children.append(step.name)
        self.roots = [name for name, degree in graph.in_degree() if degree == 0]
        self.required_by_dataset_viewer = [
            step.name for step in self.steps.values() if step.required_by_dataset_viewer
        ]
        self.topologically_ordered_steps = list(nx.topological_sort(graph))
        self.provide_dataset_config_names = [
            step.name for step in self.steps.values() if step.provides_dataset_config_names
        ]
        self.provide_config_split_names = [
            step.name for step in self.steps.values() if step.provides_config_split_names
        ]

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
        return [self.get_step(name) for name in self.roots]

    def get_steps_required_by_dataset_viewer(self) -> List[ProcessingStep]:
        """Get the steps required by the dataset viewer."""
        return [self.get_step(name) for name in self.required_by_dataset_viewer]

    def get_steps_that_provide_config_split_names(self) -> List[ProcessingStep]:
        """Get the steps that provide a config's split names."""
        return [self.get_step(name) for name in self.provide_config_split_names]

    def get_steps_that_provide_dataset_config_names(self) -> List[ProcessingStep]:
        """Get the steps that provide a dataset's config names."""
        return [self.get_step(name) for name in self.provide_dataset_config_names]

    def get_topologically_ordered_steps(self) -> List[ProcessingStep]:
        """Get the steps, ordered topologically."""
        return [self.get_step(name) for name in self.topologically_ordered_steps]
