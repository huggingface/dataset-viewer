# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from __future__ import annotations

import contextlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Priority, Queue
from libcommon.simple_cache import (
    CacheEntryMetadata,
    DoesNotExist,
    get_best_response,
    get_response,
    get_response_metadata,
)

# TODO: use the term Artifact elsewhere in the code (a Step should produce one or several Artifacts, depending on the
# input level: one, one per dataset, one per config, or one per split)
# A job and a cache entry is related to an Artifact, not to a Step
# TODO: assets, cached_assets, parquet files
# TODO: obsolete/dangling cache entries and jobs
# TODO: report, show in endpoint
# TODO: plan what to do (backfill: create job, delete cache entries, delete assets)
# TODO: add git version
# TODO: add details about jobs (priority, force, status, times)

HARD_CODED_CONFIG_NAMES_CACHE_KIND = "/config-names"
HARD_CODED_SPLIT_NAMES_FROM_STREAMING_CACHE_KIND = "/split-names-from-streaming"
HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND = "/split-names-from-dataset-info"


def fetch_config_names(dataset: str) -> List[str]:
    """Fetch the list of config names from the database."""
    config_names = []

    response = get_response(HARD_CODED_CONFIG_NAMES_CACHE_KIND, dataset=dataset, config=None, split=None)
    for config_name_item in response["content"]["config_names"]:
        config_name = config_name_item["config"]
        if not isinstance(config_name, str):
            raise ValueError(f"Invalid config name: {config_name}, type should be str, got: {type(config_name)}")
        config_names.append(config_name)
    return config_names


def fetch_split_names(dataset: str, config: str) -> List[str]:
    """Fetch the list of config names from the database."""
    split_names = []

    best_response = get_best_response(
        [HARD_CODED_SPLIT_NAMES_FROM_DATASET_INFO_CACHE_KIND, HARD_CODED_SPLIT_NAMES_FROM_STREAMING_CACHE_KIND],
        dataset=dataset,
        config=config,
        split=None,
    )
    for split_name_item in best_response.response["content"]["split_names"]:
        split_name = split_name_item["split"]
        if not isinstance(split_name, str):
            raise ValueError(f"Invalid split name: {split_name}, type should be str, got: {type(split_name)}")
        split_names.append(split_name)
    return split_names


@dataclass
class JobState:
    """The state of a job for a given input."""

    dataset: str
    config: Optional[str]
    split: Optional[str]
    job_type: str
    is_in_process: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_in_process = Queue().is_job_in_process(
            job_type=self.job_type, dataset=self.dataset, config=self.config, split=self.split
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "is_in_process": self.is_in_process,
        }


ERROR_CODES_TO_RETRY: List[str] = []


@dataclass
class CacheState:
    """The state of a cache entry for a given input."""

    dataset: str
    config: Optional[str]
    split: Optional[str]
    cache_kind: str
    cache_entry_metadata: Optional[CacheEntryMetadata] = field(init=False)
    exists: bool = field(init=False)
    is_success: bool = field(init=False)

    def __post_init__(self) -> None:
        self.cache_entry_metadata = None
        with contextlib.suppress(DoesNotExist):
            self.cache_entry_metadata = get_response_metadata(
                kind=self.cache_kind, dataset=self.dataset, config=self.config, split=self.split
            )
        """Whether the cache entry exists."""
        self.exists = self.cache_entry_metadata is not None
        self.is_success = self.cache_entry_metadata is not None and self.cache_entry_metadata["http_status"] < 400

    def as_dict(self) -> Dict[str, Any]:
        return {
            "exists": self.exists,
            "is_success": self.is_success,
        }

    def is_empty(self) -> bool:
        return self.cache_entry_metadata is None

    def is_error_to_retry(self) -> bool:
        return self.cache_entry_metadata is not None and (
            self.cache_entry_metadata["http_status"] >= 400
            and self.cache_entry_metadata["error_code"] in ERROR_CODES_TO_RETRY
        )

    def is_older_than(self, other: "CacheState") -> bool:
        if self.cache_entry_metadata is None or other.cache_entry_metadata is None:
            return False
        return self.cache_entry_metadata["updated_at"] < other.cache_entry_metadata["updated_at"]

    # TODO: old git revision
    # TODO: old job_runner_version


@dataclass
class ArtifactState:
    """The state of an artifact."""

    step: ProcessingStep
    dataset: str
    config: Optional[str]
    split: Optional[str]

    job_state: JobState = field(init=False)
    cache_state: CacheState = field(init=False)

    def __post_init__(self) -> None:
        if self.step.input_type == "dataset":
            if self.config is not None or self.split is not None:
                raise ValueError("Step input type is dataset, but config or split is not None")
        elif self.step.input_type == "config":
            if self.config is None or self.split is not None:
                raise ValueError("Step input type is config, but config is None or split is not None")
        elif self.step.input_type == "split":
            if self.config is None or self.split is None:
                raise ValueError("Step input type is split, but config or split is None")
        else:
            raise ValueError(f"Invalid step input type: {self.step.input_type}")
        self.id = ",".join([p for p in (self.step.name, self.dataset, self.config, self.split) if p])

        self.job_state = JobState(
            job_type=self.step.job_type, dataset=self.dataset, config=self.config, split=self.split
        )
        self.cache_state = CacheState(
            cache_kind=self.step.cache_kind, dataset=self.dataset, config=self.config, split=self.split
        )

    def get_parent_artifact_states(self, parent_step: ProcessingStep) -> List["ArtifactState"]:
        if parent_step.input_type == "dataset":
            return [ArtifactState(step=parent_step, dataset=self.dataset, config=None, split=None)]
        elif parent_step.input_type == "config":
            return (
                [
                    ArtifactState(
                        step=parent_step,
                        dataset=self.dataset,
                        config=self.config,
                        split=None,
                    )
                ]
                if self.step.input_type in ["config", "split"]
                else []
                # ^ fan-in: config->dataset. For now, we don't return the list of parent artifact states in that case
            )
        else:
            return (
                [
                    ArtifactState(
                        step=parent_step,
                        dataset=self.dataset,
                        config=self.config,
                        split=self.split,
                    )
                ]
                if self.step.input_type == "split"
                else []
                # ^ fan-in: split->config, or split->dataset. For now, we don't return the list of parent artifact
                #  states in that case
            )

    def get_all_parents_artifact_states(self) -> List[List["ArtifactState"]]:
        return [self.get_parent_artifact_states(parent_step) for parent_step in self.step.parents]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_state": self.job_state.as_dict(),
            "cache_state": self.cache_state.as_dict(),
        }


@dataclass
class SplitState:
    """The state of a split."""

    dataset: str
    config: str
    split: str
    processing_graph: ProcessingGraph

    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)

    def __post_init__(self) -> None:
        self.artifact_state_by_step = {
            step.name: ArtifactState(step=step, dataset=self.dataset, config=self.config, split=self.split)
            for step in self.processing_graph.steps.values()
            if step.input_type == "split"
        }

    def as_dict(self) -> Dict[str, Any]:
        return {
            "split": self.split,
            "artifact_states": [artifact_state.as_dict() for artifact_state in self.artifact_state_by_step.values()],
        }


@dataclass
class ConfigState:
    """The state of a config."""

    dataset: str
    config: str
    processing_graph: ProcessingGraph

    split_names: List[str] = field(init=False)
    split_states: List[SplitState] = field(init=False)
    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)

    def __post_init__(self) -> None:
        self.artifact_state_by_step = {
            step.name: ArtifactState(step=step, dataset=self.dataset, config=self.config, split=None)
            for step in self.processing_graph.steps.values()
            if step.input_type == "config"
        }

        try:
            self.split_names = fetch_split_names(self.dataset, self.config)
        except Exception:
            self.split_names = []

        self.split_states = [
            SplitState(self.dataset, self.config, split_name, processing_graph=self.processing_graph)
            for split_name in self.split_names
        ]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config,
            "split_states": [split_state.as_dict() for split_state in self.split_states],
            "artifact_states": [artifact_state.as_dict() for artifact_state in self.artifact_state_by_step.values()],
        }


@dataclass
class CacheStatus:
    blocked_by_parent: Dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_outdated_by_parent: Dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_empty: Dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_error_to_retry: Dict[str, ArtifactState] = field(default_factory=dict)
    up_to_date: Dict[str, ArtifactState] = field(default_factory=dict)


@dataclass
class QueueStatus:
    in_process: Dict[str, ArtifactState] = field(default_factory=dict)


@dataclass
class Task(ABC):
    artifact_state: ArtifactState

    id: str = field(init=False)

    @abstractmethod
    def run(self) -> None:
        pass


@dataclass
class CreateJobTask(Task):
    force: bool
    priority: Priority

    def __post_init__(self) -> None:
        self.id = f"CreateJob[{self.artifact_state.id}]"

    def run(self) -> None:
        Queue().upsert_job(
            job_type=self.artifact_state.step.job_type,
            dataset=self.artifact_state.dataset,
            config=self.artifact_state.config,
            split=self.artifact_state.split,
            force=self.force,
            priority=self.priority,
        )


@dataclass
class DeleteJobTask(Task):
    def __post_init__(self) -> None:
        self.id = f"DeleteJob[{self.artifact_state.id}]"

    def run(self) -> None:
        # TODO: the started jobs are also canceled: we need to ensure the job runners will
        # not try to update the cache when they finish
        Queue().cancel_pending_jobs(
            job_type=self.artifact_state.step.job_type,
            dataset=self.artifact_state.dataset,
            config=self.artifact_state.config,
            split=self.artifact_state.split,
        )


@dataclass
class Plan:
    tasks: List[Task] = field(default_factory=list)

    def add(self, task: Task) -> None:
        self.tasks.append(task)

    def run(self) -> None:
        for task in self.tasks:
            task.run()


@dataclass
class DatasetState:
    """The state of a dataset."""

    dataset: str
    processing_graph: ProcessingGraph

    config_names: List[str] = field(init=False)
    config_states: List[ConfigState] = field(init=False)
    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)
    cache_status: CacheStatus = field(init=False)
    queue_status: QueueStatus = field(init=False)
    plan: Plan = field(init=False)

    def __post_init__(self) -> None:
        self.artifact_state_by_step = {
            step.name: ArtifactState(step=step, dataset=self.dataset, config=None, split=None)
            for step in self.processing_graph.steps.values()
            if step.input_type == "dataset"
        }
        try:
            self.config_names = fetch_config_names(self.dataset)
        except Exception:
            self.config_names = []
        self.config_states = [
            ConfigState(dataset=self.dataset, config=config_name, processing_graph=self.processing_graph)
            for config_name in self.config_names
        ]
        self.cache_status = self._get_cache_status()
        self.queue_status = self._get_queue_status()
        self.plan = self._create_plan()

    def _get_artifact_states_for_step(self, step: ProcessingStep) -> List[ArtifactState]:
        if step.input_type == "dataset":
            artifact_states = [self.artifact_state_by_step[step.name]]
        elif step.input_type == "config":
            artifact_states = [config_state.artifact_state_by_step[step.name] for config_state in self.config_states]
        elif step.input_type == "split":
            artifact_states = [
                split_state.artifact_state_by_step[step.name]
                for config_state in self.config_states
                for split_state in config_state.split_states
            ]
        else:
            raise ValueError(f"Invalid input type: {step.input_type}")
        artifact_states_ids = {artifact_state.id for artifact_state in artifact_states}
        if len(artifact_states_ids) != len(artifact_states):
            raise ValueError(f"Duplicate artifact states for step {step.name}")
        return artifact_states

    def _get_cache_status(self) -> CacheStatus:
        cache_status = CacheStatus()

        for step in self.processing_graph.topologically_ordered_steps:
            artifact_states = self._get_artifact_states_for_step(step)
            for artifact_state in artifact_states:
                # (fan-in steps: config -> dataset, split -> config, split -> dataset)
                # what should we do? always recompute? test the progress?
                all_not_none_parents_artifact_states = [
                    x for x in artifact_state.get_all_parents_artifact_states() if x is not None
                ]

                # blocked by a parent?
                if any(
                    parent_artifact_state.id not in cache_status.up_to_date
                    for all_parents_artifact_state in all_not_none_parents_artifact_states
                    for parent_artifact_state in all_parents_artifact_state
                ):
                    cache_status.blocked_by_parent[artifact_state.id] = artifact_state
                    continue

                # any of the parents is more recent?
                if any(
                    artifact_state.cache_state.is_older_than(parent_artifact_state.cache_state)
                    for all_parents_artifact_state in all_not_none_parents_artifact_states
                    for parent_artifact_state in all_parents_artifact_state
                ):
                    cache_status.cache_is_outdated_by_parent[artifact_state.id] = artifact_state
                    continue

                # is empty?
                if artifact_state.cache_state.is_empty():
                    cache_status.cache_is_empty[artifact_state.id] = artifact_state
                    continue

                # is an error that can be retried?
                if artifact_state.cache_state.is_error_to_retry():
                    cache_status.cache_is_error_to_retry[artifact_state.id] = artifact_state
                    continue

                # ok
                cache_status.up_to_date[artifact_state.id] = artifact_state

        return cache_status

    def _get_queue_status(self) -> QueueStatus:
        queue_status = QueueStatus()

        for step in self.processing_graph.topologically_ordered_steps:
            artifact_states = self._get_artifact_states_for_step(step)
            for artifact_state in artifact_states:
                if artifact_state.job_state.is_in_process:
                    queue_status.in_process[artifact_state.id] = artifact_state

        return queue_status

    def _create_plan(self) -> Plan:
        plan = Plan()
        remaining_in_process_artifact_state_ids = list(self.queue_status.in_process.keys())
        artifact_states = (
            list(self.cache_status.cache_is_empty.values())
            + list(self.cache_status.cache_is_error_to_retry.values())
            + list(self.cache_status.cache_is_outdated_by_parent.values())
        )
        for artifact_state in artifact_states:
            if artifact_state.id in remaining_in_process_artifact_state_ids:
                # the job already exists
                remaining_in_process_artifact_state_ids.remove(artifact_state.id)
                continue
            plan.add(CreateJobTask(artifact_state=artifact_state, force=True, priority=Priority.LOW))
        for artifact_state_id in remaining_in_process_artifact_state_ids:
            plan.add(DeleteJobTask(artifact_state=self.queue_status.in_process[artifact_state_id]))
        return plan

    def backfill(self) -> None:
        self.plan.run()

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "config_states": [config_state.as_dict() for config_state in self.config_states],
            "artifact_states": [artifact_state.as_dict() for artifact_state in self.artifact_state_by_step.values()],
        }
