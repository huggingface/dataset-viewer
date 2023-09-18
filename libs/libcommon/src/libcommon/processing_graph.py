# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, TypedDict, Union, get_args

import networkx as nx

from libcommon.constants import (
    DEFAULT_DIFFICULTY,
    DEFAULT_INPUT_TYPE,
    DEFAULT_JOB_RUNNER_VERSION,
)
from libcommon.utils import inputs_to_string

InputType = Literal["dataset", "config", "split"]
# ^ note that for now, the "dataset" input type means: dataset + git revision


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
    triggered_by: Union[list[str], str, None]
    enables_preview: Literal[True]
    enables_viewer: Literal[True]
    enables_search: Literal[True]
    job_runner_version: int
    provides_dataset_config_names: bool
    provides_config_split_names: bool
    provides_config_parquet: bool
    provides_config_parquet_metadata: bool
    difficulty: int


ProcessingGraphSpecification = Mapping[str, ProcessingStepSpecification]


class ProcessingStepDoesNotExist(Exception):
    pass


@dataclass
class ProcessingStep:
    """A dataset processing step.

    Attributes:
        name (str): The processing step name.
        input_type (InputType): The input type ('dataset', 'config' or 'split').
        job_runner_version (int): The version of the job runner to use to compute the response.

    Getters:
        cache_kind (str): The cache kind (ie. the key in the cache).
        job_type (str): The job type (ie. the job to run to compute the response).
    """

    name: str
    input_type: InputType
    job_runner_version: int
    difficulty: int

    cache_kind: str = field(init=False)
    job_type: str = field(init=False)

    def __post_init__(self) -> None:
        self.cache_kind = self.name
        self.job_type = self.name

    def copy(self) -> ProcessingStep:
        """Copy the processing step.

        Returns:
            ProcessingStep: The copy of the processing step.
        """
        return ProcessingStep(
            name=self.name,
            input_type=self.input_type,
            job_runner_version=self.job_runner_version,
            difficulty=self.difficulty,
        )


def get_triggered_by_as_list(triggered_by: Union[list[str], str, None]) -> list[str]:
    if triggered_by is None:
        return []
    return [triggered_by] if isinstance(triggered_by, str) else triggered_by


def copy_processing_steps_list(processing_steps: list[ProcessingStep]) -> list[ProcessingStep]:
    return [processing_step.copy() for processing_step in processing_steps]


@dataclass
class ProcessingGraph:
    """A graph of processing steps.

    The processing steps can have multiple parents, and multiple children (next processing steps, found automatically
      by traversing the graph).
    The graph can have multiple roots.

    Args:
        processing_graph_specification (ProcessingGraphSpecification): The specification of the graph.

    Raises:
        ValueError: If the graph is not a DAG.
        ValueError: If a processing step provides dataset config names but its input type is not 'dataset', or if a
          processing step provides config split names but its input type is not 'config'.
        ValueError: If a root processing step (ie. a processing step with no parent) is not a dataset processing step.
    """

    processing_graph_specification: ProcessingGraphSpecification

    _nx_graph: nx.DiGraph = field(init=False)
    _processing_steps: Mapping[str, ProcessingStep] = field(init=False)
    _processing_step_names_by_input_type: Mapping[InputType, list[str]] = field(init=False)
    _first_processing_steps: list[ProcessingStep] = field(init=False)
    _processing_steps_enables_preview: list[ProcessingStep] = field(init=False)
    _processing_steps_enables_viewer: list[ProcessingStep] = field(init=False)
    _processing_steps_enables_search: list[ProcessingStep] = field(init=False)
    _config_split_names_processing_steps: list[ProcessingStep] = field(init=False)
    _config_parquet_processing_steps: list[ProcessingStep] = field(init=False)
    _config_parquet_metadata_processing_steps: list[ProcessingStep] = field(init=False)
    _dataset_config_names_processing_steps: list[ProcessingStep] = field(init=False)
    _topologically_ordered_processing_steps: list[ProcessingStep] = field(init=False)
    _alphabetically_ordered_processing_steps: list[ProcessingStep] = field(init=False)

    def __post_init__(self) -> None:
        _nx_graph = nx.DiGraph()
        _processing_steps: dict[str, ProcessingStep] = {}
        _processing_step_names_by_input_type: dict[InputType, list[str]] = {
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
            provides_config_parquet = specification.get("provides_config_parquet", False)
            if provides_config_parquet and input_type != "config":
                raise ValueError(f"Processing step {name} provides config parquet but its input type is {input_type}.")
            provides_config_parquet_metadata = specification.get("provides_config_parquet_metadata", False)
            if provides_config_parquet_metadata and input_type != "config":
                raise ValueError(
                    f"Processing step {name} provides config parquet metadata but its input type is {input_type}."
                )
            if (
                _nx_graph.has_node(name)
                or name in _processing_steps
                or name in _processing_step_names_by_input_type[input_type]
            ):
                raise ValueError(f"Processing step {name} is defined twice.")
            _nx_graph.add_node(
                name,
                enables_preview=specification.get("enables_preview", False),
                enables_viewer=specification.get("enables_viewer", False),
                enables_search=specification.get("enables_search", False),
                provides_dataset_config_names=provides_dataset_config_names,
                provides_config_split_names=provides_config_split_names,
                provides_config_parquet=provides_config_parquet,
                provides_config_parquet_metadata=provides_config_parquet_metadata,
            )
            _processing_steps[name] = ProcessingStep(
                name=name,
                input_type=input_type,
                job_runner_version=specification.get("job_runner_version", DEFAULT_JOB_RUNNER_VERSION),
                difficulty=specification.get("difficulty", DEFAULT_DIFFICULTY),
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
        self._first_processing_steps = [
            self._processing_steps[processing_step_name]
            for processing_step_name, degree in _nx_graph.in_degree()
            if degree == 0
        ]
        if any(processing_step.input_type != "dataset" for processing_step in self._first_processing_steps):
            raise ValueError("The first processing steps must be dataset-level. The graph state is incoherent.")
        self._processing_steps_enables_preview = [
            self._processing_steps[processing_step_name]
            for (processing_step_name, required) in _nx_graph.nodes(data="enables_preview")
            if required
        ]
        self._processing_steps_enables_viewer = [
            self._processing_steps[processing_step_name]
            for (processing_step_name, required) in _nx_graph.nodes(data="enables_viewer")
            if required
        ]
        self._processing_steps_enables_search = [
            self._processing_steps[processing_step_name]
            for (processing_step_name, required) in _nx_graph.nodes(data="enables_search")
            if required
        ]
        self._config_parquet_processing_steps = [
            self._processing_steps[processing_step_name]
            for (processing_step_name, provides) in _nx_graph.nodes(data="provides_config_parquet")
            if provides
        ]
        self._config_parquet_metadata_processing_steps = [
            self._processing_steps[processing_step_name]
            for (processing_step_name, provides) in _nx_graph.nodes(data="provides_config_parquet_metadata")
            if provides
        ]
        self._config_split_names_processing_steps = [
            self._processing_steps[processing_step_name]
            for (processing_step_name, provides) in _nx_graph.nodes(data="provides_config_split_names")
            if provides
        ]
        self._dataset_config_names_processing_steps = [
            self.get_processing_step(processing_step_name)
            for (processing_step_name, provides) in _nx_graph.nodes(data="provides_dataset_config_names")
            if provides
        ]
        self._topologically_ordered_processing_steps = [
            self.get_processing_step(processing_step_name) for processing_step_name in nx.topological_sort(_nx_graph)
        ]
        self._alphabetically_ordered_processing_steps = [
            self.get_processing_step(processing_step_name) for processing_step_name in sorted(_nx_graph.nodes())
        ]

    def get_processing_step(self, processing_step_name: str) -> ProcessingStep:
        """
        Get a processing step by its name.

        The returned processing step is a copy of the original one, so that it can be modified without affecting the
        original one.

        Args:
            processing_step_name (str): The name of the processing step

        Returns:
            ProcessingStep: The processing step
        """
        try:
            return self._processing_steps[processing_step_name].copy()
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown job type: {processing_step_name}") from e

    def get_processing_step_by_job_type(self, job_type: str) -> ProcessingStep:
        """
        Get a processing step by its job type.

        The returned processing step is a copy of the original one, so that it can be modified without affecting the
        original one.

        Args:
            job_type (str): The job type of the processing step

        Returns:
            ProcessingStep: The processing step
        """
        # for now: the job_type is just an alias for the processing step name
        return self.get_processing_step(job_type)

    def get_children(self, processing_step_name: str) -> list[ProcessingStep]:
        """
        Get the list of children processing steps

        The children processing steps are the ones that will be triggered at the end of the processing step.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Args:
            processing_step_name (str): The name of the processing step

        Returns:
            list[ProcessingStep]: The list of children processing steps (successors)

        Raises:
            ProcessingStepDoesNotExist: If the processing step is not in the graph
        """
        try:
            return [
                self.get_processing_step(successor) for successor in self._nx_graph.successors(processing_step_name)
            ]
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown processing step: {processing_step_name}") from e

    def get_parents(self, processing_step_name: str) -> list[ProcessingStep]:
        """
        Get the list of parents processing steps

        The parent processing steps are the ones that trigger the processing step.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Args:
            processing_step_name (str): The name of the processing step

        Returns:
            list[ProcessingStep]: The list of parent processing steps (predecessors)

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

    def get_ancestors(self, processing_step_name: str) -> list[ProcessingStep]:
        """
        Get the list of ancestors processing steps

        The ancestor processing steps are the ones that trigger the processing step, directly or not.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Args:
            processing_step_name (str): The name of the processing step

        Returns:
            list[ProcessingStep]: The list of ancestor processing steps

        Raises:
            ProcessingStepDoesNotExist: If the processing step is not in the graph
        """
        try:
            return [
                self.get_processing_step(ancestor) for ancestor in nx.ancestors(self._nx_graph, processing_step_name)
            ]
        except nx.NetworkXError as e:
            raise ProcessingStepDoesNotExist(f"Unknown processing step: {processing_step_name}") from e

    def get_first_processing_steps(self) -> list[ProcessingStep]:
        """
        Get the first processing steps.

        The first processing steps are the ones that don't have a previous step. This means that they will be computed
        first when a dataset is updated. Their input type is always "dataset".

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            list[ProcessingStep]: The list of first processing steps
        """
        return copy_processing_steps_list(self._first_processing_steps)

    def get_processing_steps_enables_preview(self) -> list[ProcessingStep]:
        """
        Get the processing steps that enable the dataset preview (first rows).

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            list[ProcessingStep]: The list of processing steps that enable the dataset preview
        """
        return copy_processing_steps_list(self._processing_steps_enables_preview)

    def get_processing_steps_enables_viewer(self) -> list[ProcessingStep]:
        """
        Get the processing steps that enable the dataset viewer (all rows).

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            list[ProcessingStep]: The list of processing steps that enable the dataset viewer
        """
        return copy_processing_steps_list(self._processing_steps_enables_viewer)

    def get_processing_steps_enables_search(self) -> list[ProcessingStep]:
        """
        Get the processing steps that enable the dataset split search.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            list[ProcessingStep]: The list of processing steps that enable the dataset viewer
        """
        return copy_processing_steps_list(self._processing_steps_enables_search)

    def get_config_parquet_processing_steps(self) -> list[ProcessingStep]:
        """
        Get the processing steps that provide a config's parquet response.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            list[ProcessingStep]: The list of processing steps that provide a config's parquet response
        """
        return copy_processing_steps_list(self._config_parquet_processing_steps)

    def get_config_parquet_metadata_processing_steps(self) -> list[ProcessingStep]:
        """
        Get the processing steps that provide a config's parquet metadata response.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            list[ProcessingStep]: The list of processing steps that provide a config's parquet response
        """
        return copy_processing_steps_list(self._config_parquet_metadata_processing_steps)

    def get_config_split_names_processing_steps(self) -> list[ProcessingStep]:
        """
        Get the processing steps that provide a config's split names.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            list[ProcessingStep]: The list of processing steps that provide a config's split names
        """
        return copy_processing_steps_list(self._config_split_names_processing_steps)

    def get_dataset_config_names_processing_steps(self) -> list[ProcessingStep]:
        """
        Get the processing steps that provide a dataset's config names.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            list[ProcessingStep]: The list of processing steps that provide a dataset's config names
        """
        return copy_processing_steps_list(self._dataset_config_names_processing_steps)

    def get_topologically_ordered_processing_steps(self) -> list[ProcessingStep]:
        """
        Get the processing steps, ordered topologically.

        This means that the first processing steps are the ones that don't have a previous step, and that the last
        processing steps are the ones that don't have a next step.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            list[ProcessingStep]: The list of processing steps
        """
        return copy_processing_steps_list(self._topologically_ordered_processing_steps)

    def get_alphabetically_ordered_processing_steps(self) -> list[ProcessingStep]:
        """
        Get the processing steps, ordered alphabetically by the name of the processing steps.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Returns:
            list[ProcessingStep]: The list of processing steps
        """
        return copy_processing_steps_list(self._alphabetically_ordered_processing_steps)

    def get_processing_steps(
        self, order: Optional[Literal["alphabetical", "topological"]] = None
    ) -> list[ProcessingStep]:
        """
        Get the processing steps.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Args:
            order (Optional[Literal["alphabetical", "topological"]], optional): The order in which to return the
              processing steps. If None, the order is alphabetical. Defaults to None.

        Returns:
            list[ProcessingStep]: The list of processing steps
        """
        if order == "topological":
            return self.get_topologically_ordered_processing_steps()
        # default
        return self.get_alphabetically_ordered_processing_steps()

    def get_input_type_processing_steps(self, input_type: InputType = "dataset") -> list[ProcessingStep]:
        """
        Get the processing steps of input type `input_type`, in an undefined order.

        The returned processing steps are copies of the original ones, so that they can be modified without affecting
        the original ones.

        Args:
            input_type (InputType, optional): The input type. Defaults to "dataset".

        Returns:
            list[ProcessingStep]: The list of processing steps
        """
        return [
            self.get_processing_step(processing_step_name)
            for processing_step_name in self._processing_step_names_by_input_type[input_type]
        ]


@dataclass
class Artifact:
    """An artifact."""

    processing_step: ProcessingStep
    dataset: str
    revision: str
    config: Optional[str]
    split: Optional[str]

    id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.processing_step.input_type == "dataset":
            if self.config is not None or self.split is not None:
                raise ValueError("Step input type is dataset, but config or split is not None")
        elif self.processing_step.input_type == "config":
            if self.config is None or self.split is not None:
                raise ValueError("Step input type is config, but config is None or split is not None")
        elif self.processing_step.input_type == "split":
            if self.config is None or self.split is None:
                raise ValueError("Step input type is split, but config or split is None")
        else:
            raise ValueError(f"Invalid step input type: {self.processing_step.input_type}")
        self.id = Artifact.get_id(
            dataset=self.dataset,
            revision=self.revision,
            config=self.config,
            split=self.split,
            processing_step_name=self.processing_step.name,
        )

    @staticmethod
    def get_id(
        dataset: str,
        revision: str,
        config: Optional[str],
        split: Optional[str],
        processing_step_name: str,
    ) -> str:
        return inputs_to_string(
            dataset=dataset,
            revision=revision,
            config=config,
            split=split,
            prefix=processing_step_name,
        )

    @staticmethod
    def parse_id(id: str) -> tuple[str, str, Optional[str], Optional[str], str]:
        parts = id.split(",")
        prefix = parts[0]
        parts = parts[1:]
        dataset = parts[0]
        revision = parts[1]
        parts = parts[2:]
        config = None
        split = None
        if len(parts) > 1:
            config = parts[1]
            if len(parts) > 2:
                split = parts[2]
        return dataset, revision, config, split, prefix
