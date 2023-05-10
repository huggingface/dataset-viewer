# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from __future__ import annotations

import contextlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.queue import Priority, Queue, Status
from libcommon.simple_cache import (
    CacheEntryMetadata,
    DoesNotExist,
    get_best_response,
    get_response_metadata,
)
from libcommon.utils import inputs_to_string

# TODO: use the term Artifact elsewhere in the code (a Step should produce one or several Artifacts, depending on the
# input level: one, one per dataset, one per config, or one per split)
# A job and a cache entry is related to an Artifact, not to a Step
# TODO: assets, cached_assets, parquet files
# TODO: obsolete/dangling cache entries and jobs


def fetch_names(
    dataset: str, config: Optional[str], cache_kinds: List[str], names_field: str, name_field: str
) -> List[str]:
    """Fetch a list of names from the database."""
    names = []

    best_response = get_best_response(kinds=cache_kinds, dataset=dataset, config=config)
    for name_item in best_response.response["content"][names_field]:
        name = name_item[name_field]
        if not isinstance(name, str):
            raise ValueError(f"Invalid name: {name}, type should be str, got: {type(name)}")
        names.append(name)
    return names


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


@dataclass
class CacheState:
    """The state of a cache entry for a given input."""

    dataset: str
    config: Optional[str]
    split: Optional[str]
    cache_kind: str
    error_codes_to_retry: Optional[List[str]] = None
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

    def is_empty(self) -> bool:
        return self.cache_entry_metadata is None

    def is_error_to_retry(self) -> bool:
        return (
            self.error_codes_to_retry is not None
            and self.cache_entry_metadata is not None
            and (
                self.cache_entry_metadata["http_status"] >= 400
                and self.cache_entry_metadata["error_code"] in self.error_codes_to_retry
            )
        )

    def is_older_than(self, other: "CacheState") -> bool:
        if self.cache_entry_metadata is None or other.cache_entry_metadata is None:
            return False
        return self.cache_entry_metadata["updated_at"] < other.cache_entry_metadata["updated_at"]

    def is_git_revision_different_from(self, git_revision: Optional[str]) -> bool:
        return self.cache_entry_metadata is None or self.cache_entry_metadata["dataset_git_revision"] != git_revision


@dataclass
class ArtifactState:
    """The state of an artifact."""

    processing_step: ProcessingStep
    dataset: str
    config: Optional[str]
    split: Optional[str]
    error_codes_to_retry: Optional[List[str]] = None

    job_state: JobState = field(init=False)
    cache_state: CacheState = field(init=False)

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
        self.id = inputs_to_string(
            dataset=self.dataset, config=self.config, split=self.split, prefix=self.processing_step.name
        )

        self.job_state = JobState(
            job_type=self.processing_step.job_type,
            dataset=self.dataset,
            config=self.config,
            split=self.split,
        )
        self.cache_state = CacheState(
            cache_kind=self.processing_step.cache_kind,
            dataset=self.dataset,
            config=self.config,
            split=self.split,
            error_codes_to_retry=self.error_codes_to_retry,
        )

    def is_job_runner_obsolete(self) -> bool:
        if self.cache_state.cache_entry_metadata is None:
            return False
        job_runner_version = self.cache_state.cache_entry_metadata["job_runner_version"]
        if job_runner_version is None:
            return True
        return job_runner_version < self.processing_step.job_runner_version


@dataclass
class SplitState:
    """The state of a split."""

    dataset: str
    config: str
    split: str
    processing_graph: ProcessingGraph
    error_codes_to_retry: Optional[List[str]] = None

    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)

    def __post_init__(self) -> None:
        self.artifact_state_by_step = {
            processing_step.name: ArtifactState(
                processing_step=processing_step,
                dataset=self.dataset,
                config=self.config,
                split=self.split,
                error_codes_to_retry=self.error_codes_to_retry,
            )
            for processing_step in self.processing_graph.get_input_type_processing_steps(input_type="split")
        }


@dataclass
class ConfigState:
    """The state of a config."""

    dataset: str
    config: str
    processing_graph: ProcessingGraph
    error_codes_to_retry: Optional[List[str]] = None

    split_names: List[str] = field(init=False)
    split_states: List[SplitState] = field(init=False)
    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)

    def __post_init__(self) -> None:
        self.artifact_state_by_step = {
            processing_step.name: ArtifactState(
                processing_step=processing_step,
                dataset=self.dataset,
                config=self.config,
                split=None,
                error_codes_to_retry=self.error_codes_to_retry,
            )
            for processing_step in self.processing_graph.get_input_type_processing_steps(input_type="config")
        }

        try:
            self.split_names = fetch_names(
                dataset=self.dataset,
                config=self.config,
                cache_kinds=[
                    processing_step.cache_kind
                    for processing_step in self.processing_graph.get_config_split_names_processing_steps()
                ],
                names_field="splits",
                name_field="split",
            )
        except Exception:
            self.split_names = []

        self.split_states = [
            SplitState(
                self.dataset,
                self.config,
                split_name,
                processing_graph=self.processing_graph,
                error_codes_to_retry=self.error_codes_to_retry,
            )
            for split_name in self.split_names
        ]


@dataclass
class CacheStatus:
    cache_has_different_git_revision: Dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_outdated_by_parent: Dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_empty: Dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_error_to_retry: Dict[str, ArtifactState] = field(default_factory=dict)
    cache_is_job_runner_obsolete: Dict[str, ArtifactState] = field(default_factory=dict)
    up_to_date: Dict[str, ArtifactState] = field(default_factory=dict)

    def as_response(self) -> Dict[str, List[str]]:
        return {
            "cache_has_different_git_revision": sorted(self.cache_has_different_git_revision.keys()),
            "cache_is_outdated_by_parent": sorted(self.cache_is_outdated_by_parent.keys()),
            "cache_is_empty": sorted(self.cache_is_empty.keys()),
            "cache_is_error_to_retry": sorted(self.cache_is_error_to_retry.keys()),
            "cache_is_job_runner_obsolete": sorted(self.cache_is_job_runner_obsolete.keys()),
            "up_to_date": sorted(self.up_to_date.keys()),
        }


@dataclass
class QueueStatus:
    in_process: Dict[str, ArtifactState] = field(default_factory=dict)

    def as_response(self) -> Dict[str, List[str]]:
        return {"in_process": sorted(self.in_process.keys())}


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
        self.id = f"CreateJob,{self.artifact_state.id}"

    def run(self) -> None:
        Queue().upsert_job(
            job_type=self.artifact_state.processing_step.job_type,
            dataset=self.artifact_state.dataset,
            config=self.artifact_state.config,
            split=self.artifact_state.split,
            force=self.force,
            priority=self.priority,
        )


@dataclass
class DeleteJobTask(Task):
    def __post_init__(self) -> None:
        self.id = f"DeleteJob,{self.artifact_state.id}"

    def run(self) -> None:
        # TODO: the started jobs are also canceled: we need to ensure the job runners will
        # not try to update the cache when they finish
        Queue().cancel_jobs(
            job_type=self.artifact_state.processing_step.job_type,
            dataset=self.artifact_state.dataset,
            config=self.artifact_state.config,
            split=self.artifact_state.split,
            statuses_to_cancel=[Status.WAITING, Status.STARTED],
        )


@dataclass
class Plan:
    tasks: List[Task] = field(default_factory=list)

    def add(self, task: Task) -> None:
        self.tasks.append(task)

    def run(self) -> int:
        """Run all the tasks in the plan.

        Returns:
            The number of tasks that were run.
        """
        for task in self.tasks:
            logging.debug(f"Running task {task.id}")
            task.run()
        return len(self.tasks)

    def as_response(self) -> List[str]:
        return sorted(task.id for task in self.tasks)


@dataclass
class DatasetState:
    """The state of a dataset."""

    dataset: str
    processing_graph: ProcessingGraph
    revision: Optional[str]
    error_codes_to_retry: Optional[List[str]] = None

    config_names: List[str] = field(init=False)
    config_states: List[ConfigState] = field(init=False)
    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)
    cache_status: CacheStatus = field(init=False)
    queue_status: QueueStatus = field(init=False)
    plan: Plan = field(init=False)
    should_be_backfilled: bool = field(init=False)

    def __post_init__(self) -> None:
        self.artifact_state_by_step = {
            processing_step.name: ArtifactState(
                processing_step=processing_step,
                dataset=self.dataset,
                config=None,
                split=None,
                error_codes_to_retry=self.error_codes_to_retry,
            )
            for processing_step in self.processing_graph.get_input_type_processing_steps(input_type="dataset")
        }
        try:
            self.config_names = fetch_names(
                dataset=self.dataset,
                config=None,
                cache_kinds=[
                    processing_step.cache_kind
                    for processing_step in self.processing_graph.get_dataset_config_names_processing_steps()
                ],
                names_field="config_names",
                name_field="config",
            )
        except Exception:
            self.config_names = []
        self.config_states = [
            ConfigState(
                dataset=self.dataset,
                config=config_name,
                processing_graph=self.processing_graph,
                error_codes_to_retry=self.error_codes_to_retry,
            )
            for config_name in self.config_names
        ]
        self.cache_status = self._get_cache_status()
        self.queue_status = self._get_queue_status()
        self.plan = self._create_plan()
        self.should_be_backfilled = len(self.plan.tasks) > 0

    def _get_artifact_states_for_step(
        self, processing_step: ProcessingStep, config: Optional[str] = None, split: Optional[str] = None
    ) -> List[ArtifactState]:
        """Get the artifact states for a step.

        Args:
            processing_step (ProcessingStep): the processing step
            config (str, optional): if not None, and step input type is config or split, only return the artifact
              states for this config
            split (str, optional): if not None, and step input type is split, only return the artifact states for
              this split (config must be specified)

        Returns:
            the artifact states for the step
        """
        if processing_step.input_type == "dataset":
            artifact_states = [self.artifact_state_by_step[processing_step.name]]
        elif processing_step.input_type == "config":
            if config is None:
                artifact_states = [
                    config_state.artifact_state_by_step[processing_step.name] for config_state in self.config_states
                ]
            else:
                artifact_states = [
                    config_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.config_states
                    if config_state.config == config
                ]
        elif processing_step.input_type == "split":
            if config is None:
                artifact_states = [
                    split_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.config_states
                    for split_state in config_state.split_states
                ]
            elif split is None:
                artifact_states = [
                    split_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.config_states
                    if config_state.config == config
                    for split_state in config_state.split_states
                ]
            else:
                artifact_states = [
                    split_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.config_states
                    if config_state.config == config
                    for split_state in config_state.split_states
                    if split_state.split == split
                ]
        else:
            raise ValueError(f"Invalid input type: {processing_step.input_type}")
        artifact_states_ids = {artifact_state.id for artifact_state in artifact_states}
        if len(artifact_states_ids) != len(artifact_states):
            raise ValueError(f"Duplicate artifact states for processing_step {processing_step}")
        return artifact_states

    def _get_cache_status(self) -> CacheStatus:
        cache_status = CacheStatus()

        for processing_step in self.processing_graph.get_topologically_ordered_processing_steps():
            # Every step can have one or multiple artifacts, for example config-level steps have one artifact per
            # config
            artifact_states = self._get_artifact_states_for_step(processing_step)
            for artifact_state in artifact_states:
                # any of the parents is more recent?
                if any(
                    artifact_state.cache_state.is_older_than(parent_artifact_state.cache_state)
                    for parent_step in self.processing_graph.get_parents(processing_step.name)
                    for parent_artifact_state in self._get_artifact_states_for_step(
                        processing_step=parent_step,
                        config=artifact_state.config,
                        split=artifact_state.split,
                    )
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

                # was created with an obsolete version of the job runner?
                if artifact_state.is_job_runner_obsolete():
                    cache_status.cache_is_job_runner_obsolete[artifact_state.id] = artifact_state
                    continue

                # has a different git revision from the dataset current revision?
                if artifact_state.cache_state.is_git_revision_different_from(self.revision):
                    cache_status.cache_has_different_git_revision[artifact_state.id] = artifact_state
                    continue

                # ok
                cache_status.up_to_date[artifact_state.id] = artifact_state

        return cache_status

    def _get_queue_status(self) -> QueueStatus:
        queue_status = QueueStatus()

        for processing_step in self.processing_graph.get_topologically_ordered_processing_steps():
            artifact_states = self._get_artifact_states_for_step(processing_step)
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
            + list(self.cache_status.cache_is_job_runner_obsolete.values())
            + list(self.cache_status.cache_has_different_git_revision.values())
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

    def backfill(self) -> int:
        """Backfill the cache.

        Returns:
            The number of jobs created.
        """
        return self.plan.run()

    def as_response(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "cache_status": self.cache_status.as_response(),
            "queue_status": self.queue_status.as_response(),
            "plan": self.plan.as_response(),
        }
