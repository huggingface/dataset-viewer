# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    TypedDict,
    Union,
    get_args,
)

import networkx as nx

from libcommon.constants import DEFAULT_INPUT_TYPE, DEFAULT_JOB_RUNNER_VERSION

InputType = Literal["dataset", "config", "split"]


def guard_input_type(x: Any) -> InputType:
    if x == "dataset":
        return "dataset"
    elif x == "config":
        return "config"
    elif x == "split":
        return "split"
    if x in get_args(InputType):
        raise RuntimeError(f"Value {x} should be included in the literal values")
    raise ValueError(f"Invalid input type: {x}")


def guard_int(x: Any) -> int:
    if isinstance(x, int):
        return x
    raise ValueError(f"Invalid int: {x}")


class ProcessingStepSpecification(TypedDict, total=False):
    input_type: InputType
    triggered_by: Union[List[str], str, None]
    required_by_dataset_viewer: Literal[True]
    job_runner_version: int
    provides_dataset_config_names: bool
    provides_config_split_names: bool


class ProcessingStepDoesNotExist(Exception):
    pass


@dataclass
class ProcessingStep:
    name: str
    input_type: InputType
    job_runner_version: int

    cache_kind: str = field(init=False)
    job_type: str = field(init=False)

    def __post_init__(self) -> None:
        self.cache_kind = self.name
        self.job_type = self.name


#     """A dataset processing step.

#     Attributes:
#         name (str): The step name.
#         input_type (InputType): The input type ('dataset', 'config' or 'split').
#         job_runner_version (int): The version of the job runner to use to compute the response.
#         triggered_by (List[str], optional): The names of the steps that trigger this step, in no particular order.
#           If None, the step is a root. Defaults to None.
#         parents (List[str], optional): The names of the steps that trigger this step, in no particular order.
#           Defaults to [].
#         ancestors (List[str], optional): All the chain of previous steps names, even those that do not trigger the
#           step directly, in no particular order. Defaults to [].
#         children (List[str], optional): Names of the steps that will be triggered at the end of the step, in no
#           particular order. Defaults to [].
#         required_by_dataset_viewer (bool, optional): Whether the step is required by the dataset viewer. Defaults to
#           False.
#         provides_dataset_config_names (bool, optional): Whether the step provides dataset config names. Defaults to
#           False.
#         provides_config_split_names (bool, optional): Whether the step provides config split names. Defaults to
#           False.
#     Getters:
#         job_type (str): The job type (ie. the job to run to compute the response).
#         cache_kind (str): The cache kind (ie. the key in the cache).
# """


ProcessingGraphSpecification = Mapping[str, ProcessingStepSpecification]


def get_triggered_by_as_list(triggered_by: Union[List[str], str, None]) -> List[str]:
    if triggered_by is None:
        return []
    return [triggered_by] if isinstance(triggered_by, str) else triggered_by


@dataclass
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
        ValueError: If one step provides dataset config names but its input type is not 'dataset', or if one step
          provides config split names but its input type is not 'config'.
    """

    processing_graph_specification: ProcessingGraphSpecification

    _nx_graph: nx.DiGraph = field(init=False)
    _processing_steps: Mapping[str, ProcessingStep] = field(init=False)
    _processing_step_names_by_input_type: Mapping[InputType, List[str]] = field(init=False)
    # roots: List[str] = field(init=False)
    # required_by_dataset_viewer: List[str] = field(init=False)
    # topologically_ordered_steps: List[str] = field(init=False)
    # provide_dataset_config_names: List[str] = field(init=False)
    # provide_config_split_names: List[str] = field(init=False)

    def __post_init__(self) -> None:
        _nx_graph = nx.DiGraph()
        _processing_steps: Dict[str, ProcessingStep] = {}
        _processing_step_names_by_input_type: Dict[InputType, List[str]] = {
            "dataset": [],
            "config": [],
            "split": [],
        }
        for name, specification in self.processing_graph_specification.items():
            # check that the step is consistent with its specification
            input_type = guard_input_type(specification.get("input_type", DEFAULT_INPUT_TYPE))
            provides_dataset_config_names = specification.get("provides_dataset_config_names", False)
            if provides_dataset_config_names and input_type != "dataset":
                raise ValueError(
                    f"Processing step {name} provides dataset config names but its input type is {input_type}."
                )
            provides_config_split_names = specification.get("provides_config_split_names", False)
            if provides_config_split_names and input_type != "config":
                raise ValueError(
                    f"Processing step {name} provides config split names but its input type is {input_type}."
                )
            if (
                _nx_graph.has_node(name)
                or name in _processing_steps
                or name in _processing_step_names_by_input_type[input_type]
            ):
                raise ValueError(f"Processing step {name} is defined twice.")
            _nx_graph.add_node(
                name,
                required_by_dataset_viewer=specification.get("required_by_dataset_viewer", False),
                provides_dataset_config_names=provides_dataset_config_names,
                provides_config_split_names=provides_config_split_names,
            )
            _processing_steps[name] = ProcessingStep(
                name=name,
                input_type=input_type,
                job_runner_version=specification.get("job_runner_version", DEFAULT_JOB_RUNNER_VERSION),
            )
            _processing_step_names_by_input_type[input_type].append(name)
        for name, specification in self.processing_graph_specification.items():
            triggered_by = get_triggered_by_as_list(specification.get("triggered_by"))
            for processing_step_name in triggered_by:
                if not _nx_graph.has_node(processing_step_name):
                    raise ValueError(
                        f"Processing step {name} is triggered by {processing_step_name} but {processing_step_name} is"
                        " not defined."
                    )
                _nx_graph.add_edge(processing_step_name, name)
        if not nx.is_directed_acyclic_graph(_nx_graph):
            raise ValueError("The graph is not a directed acyclic graph.")

        self._nx_graph = _nx_graph
        self._processing_steps = _processing_steps
        self._processing_step_names_by_input_type = _processing_step_names_by_input_type
        # for processing_step in self._processing_steps.values():
        #     processing_step.ancestors = list(nx.ancestors(graph, processing_step.name))
        # for processing_step in self._processing_steps.values():
        #     processing_step.parents = list(graph.predecessors(processing_step.name))
        #     for parent in processing_step.parents:
        #         self.get_processing_step(parent).children.append(processing_step.name)
        # self.required_by_dataset_viewer = [node for node in self._nx_graph.nodes if node.required_by_dataset_viewer]
        # self.topologically_ordered_processing_steps = list(nx.topological_sort(self._nx_graph))
        # self.provide_dataset_config_names = [
        #     processing_step.name for processing_step in self._processing_steps.values()
        #     if processing_step.provides_dataset_config_names
        # ]
        # self.provide_config_split_names = [
        #     processing_step.name for processing_step in self._processing_steps.values()
        #     if processing_step.provides_config_split_names
        # ]

    def get_processing_step(self, processing_step_name: str) -> ProcessingStep:
        """Get a processing step by its name."""
        try:
            return self._processing_steps[processing_step_name]
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown job type: {processing_step_name}") from e

    def get_processing_step_by_job_type(self, job_type: str) -> ProcessingStep:
        # for now: the job_type is just an alias for the processing step name
        return self.get_processing_step(job_type)

    def get_children(self, processing_step_name: str) -> List[ProcessingStep]:
        """
        Get the list of children processing steps

        Args:
            processing_step_name (str): The name of the processing step

        Returns:
            List[ProcessingStep]: The list of children processing steps (successors)

        Raises:
            ProcessingStepDoesNotExist: If the processing step is not in the graph
        """
        try:
            return [
                self.get_processing_step(successor) for successor in self._nx_graph.successors(processing_step_name)
            ]
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown processing step: {processing_step_name}") from e

    def get_parents(self, processing_step_name: str) -> List[ProcessingStep]:
        """
        Get the list of parents processing steps

        Args:
            processing_step_name (str): The name of the processing step

        Returns:
            List[ProcessingStep]: The list of parent processing steps (predecessors)

        Raises:
            ProcessingStepDoesNotExist: If the processing step is not in the graph
        """
        try:
            return [
                self.get_processing_step(predecessor)
                for predecessor in self._nx_graph.predecessors(processing_step_name)
            ]
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown processing step: {processing_step_name}") from e

    def get_ancestors(self, processing_step_name: str) -> List[ProcessingStep]:
        """
        Get the list of ancestors processing steps

        Args:
            processing_step_name (str): The name of the processing step

        Returns:
            List[ProcessingStep]: The list of ancestor processing steps

        Raises:
            ProcessingStepDoesNotExist: If the processing step is not in the graph
        """
        try:
            return [
                self.get_processing_step(ancestor) for ancestor in nx.ancestors(self._nx_graph, processing_step_name)
            ]
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown processing step: {processing_step_name}") from e

    def get_first_processing_steps(self) -> List[ProcessingStep]:
        """Get the first processing steps."""
        processing_steps = [
            self.get_processing_step(name) for name, degree in self._nx_graph.in_degree() if degree == 0
        ]
        if any(processing_step.input_type != "dataset" for processing_step in processing_steps):
            raise ValueError("The first processing steps must be dataset-level. The graph state is incoherent.")
        return processing_steps

    def get_processing_steps_required_by_dataset_viewer(self) -> List[ProcessingStep]:
        """Get the processing steps required by the dataset viewer."""
        return [
            self.get_processing_step(name)
            for (name, required) in self._nx_graph.nodes(data="required_by_dataset_viewer")
            if required
        ]

    def get_config_split_names_processing_steps(self) -> List[ProcessingStep]:
        """Get the processing steps that provide a config's split names."""
        return [
            self.get_processing_step(name)
            for (name, provides) in self._nx_graph.nodes(data="provides_config_split_names")
            if provides
        ]

    def get_dataset_config_names_processing_steps(self) -> List[ProcessingStep]:
        """Get the processing steps that provide a dataset's config names."""
        return [
            self.get_processing_step(name)
            for (name, provides) in self._nx_graph.nodes(data="provides_dataset_config_names")
            if provides
        ]

    def get_topologically_ordered_processing_steps(self) -> List[ProcessingStep]:
        """Get the processing steps, ordered topologically."""
        return [self.get_processing_step(name) for name in nx.topological_sort(self._nx_graph)]

    def get_alphabetically_ordered_processing_steps(self) -> List[ProcessingStep]:
        """Get the processing steps, ordered alphabetically."""
        return [self.get_processing_step(name) for name in sorted(self._nx_graph.nodes())]

    def get_processing_steps(
        self, order: Optional[Literal["alphabetical", "topological"]] = None
    ) -> List[ProcessingStep]:
        """Get the processing steps.

        Args:
            order (Optional[Literal["alphabetical", "topological"]], optional): The order in which to return the
              processing steps. If None, the order is alphabetical. Defaults to None.

        Returns:
            List[str]: The list of processing steps
        """
        if order == "topological":
            return self.get_topologically_ordered_processing_steps()
        # default
        return self.get_alphabetically_ordered_processing_steps()

    def get_input_type_processing_steps(self, input_type: InputType = "dataset") -> List[ProcessingStep]:
        """Get the processing steps of input type `input_type`, in an undefined order."""
        return [
            self.get_processing_step(processing_step_name)
            for processing_step_name in self._processing_step_names_by_input_type[input_type]
        ]
