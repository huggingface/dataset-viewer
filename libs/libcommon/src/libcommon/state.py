# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.prometheus import StepProfiler
from libcommon.queue import Queue
from libcommon.simple_cache import (
    CacheEntryMetadata,
    get_best_response,
    get_cache_entries_df,
)
from libcommon.utils import Priority, Status, inputs_to_string

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


DFIndex = Union[pd.Index, pd.MultiIndex]


@dataclass
class JobState:
    """The state of a job for a given input."""

    dataset: str
    revision: str
    config: Optional[str]
    split: Optional[str]
    job_type: str
    pending_jobs_df: pd.DataFrame

    valid_pending_jobs_df: pd.DataFrame = field(
        init=False
    )  # contains at most one row (but the logic does not depend on it)
    is_in_process: bool = field(init=False)

    def __post_init__(self) -> None:
        self.valid_pending_jobs_df = self.pending_jobs_df.sort_values(
            ["status", "priority", "created_at"], ascending=[False, False, True]
        ).head(1)
        # ^ only keep the first valid job, if any, in order of priority
        self.is_in_process = not self.valid_pending_jobs_df.empty


@dataclass
class CacheState:
    """The state of a cache entry for a given input."""

    dataset: str
    config: Optional[str]
    split: Optional[str]
    cache_kind: str
    cache_entries_df: pd.DataFrame
    error_codes_to_retry: Optional[List[str]] = None

    cache_entry_metadata: Optional[CacheEntryMetadata] = field(init=False)
    exists: bool = field(init=False)
    is_success: bool = field(init=False)

    def __post_init__(self) -> None:
        if len(self.cache_entries_df) > 1:
            logging.warning(
                f"More than one cache entry found for {self.dataset}, {self.config}, {self.split}, {self.cache_kind}"
            )
        if len(self.cache_entries_df) == 0:
            self.cache_entry_metadata = None
        else:
            entry = self.cache_entries_df.iloc[0]
            self.cache_entry_metadata = CacheEntryMetadata(
                http_status=entry["http_status"],
                error_code=None if entry["error_code"] is pd.NA else entry["error_code"],
                job_runner_version=None if entry["job_runner_version"] is pd.NA else entry["job_runner_version"],
                dataset_git_revision=None if entry["dataset_git_revision"] is pd.NA else entry["dataset_git_revision"],
                updated_at=entry["updated_at"],
                progress=None if entry["progress"] is pd.NA else entry["progress"],
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


@dataclass
class ArtifactState(Artifact):
    """The state of an artifact."""

    pending_jobs_df: pd.DataFrame
    cache_entries_df: pd.DataFrame
    error_codes_to_retry: Optional[List[str]] = None

    job_state: JobState = field(init=False)
    cache_state: CacheState = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.job_state = JobState(
            job_type=self.processing_step.job_type,
            dataset=self.dataset,
            revision=self.revision,
            config=self.config,
            split=self.split,
            pending_jobs_df=self.pending_jobs_df,
        )
        self.cache_state = CacheState(
            cache_kind=self.processing_step.cache_kind,
            dataset=self.dataset,
            config=self.config,
            split=self.split,
            error_codes_to_retry=self.error_codes_to_retry,
            cache_entries_df=self.cache_entries_df,
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
    revision: str
    config: str
    split: str
    processing_graph: ProcessingGraph
    pending_jobs_df: pd.DataFrame
    cache_entries_df: pd.DataFrame
    error_codes_to_retry: Optional[List[str]] = None

    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)

    def __post_init__(self) -> None:
        self.pending_jobs_df = self.pending_jobs_df[
            (self.pending_jobs_df["dataset"] == self.dataset)
            & (self.pending_jobs_df["revision"] == self.revision)
            & (self.pending_jobs_df["config"] == self.config)
            & (self.pending_jobs_df["split"] == self.split)
        ]
        self.cache_entries_df = self.cache_entries_df[
            (self.cache_entries_df["dataset"] == self.dataset)
            & (self.cache_entries_df["config"] == self.config)
            & (self.cache_entries_df["split"] == self.split)
        ]
        # ^ safety check
        self.artifact_state_by_step = {
            processing_step.name: ArtifactState(
                processing_step=processing_step,
                dataset=self.dataset,
                revision=self.revision,
                config=self.config,
                split=self.split,
                error_codes_to_retry=self.error_codes_to_retry,
                pending_jobs_df=self.pending_jobs_df[self.pending_jobs_df["type"] == processing_step.job_type],
                cache_entries_df=self.cache_entries_df[self.cache_entries_df["kind"] == processing_step.cache_kind],
            )
            for processing_step in self.processing_graph.get_input_type_processing_steps(input_type="split")
        }


@dataclass
class ConfigState:
    """The state of a config."""

    dataset: str
    revision: str
    config: str
    processing_graph: ProcessingGraph
    pending_jobs_df: pd.DataFrame
    cache_entries_df: pd.DataFrame
    error_codes_to_retry: Optional[List[str]] = None

    split_names: List[str] = field(init=False)
    split_states: List[SplitState] = field(init=False)
    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)

    def __post_init__(self) -> None:
        self.pending_jobs_df = self.pending_jobs_df[
            (self.pending_jobs_df["dataset"] == self.dataset)
            & (self.pending_jobs_df["revision"] == self.revision)
            & (self.pending_jobs_df["config"] == self.config)
        ]
        self.cache_entries_df = self.cache_entries_df[
            (self.cache_entries_df["dataset"] == self.dataset) & (self.cache_entries_df["config"] == self.config)
        ]
        # ^ safety check
        with StepProfiler(
            method="ConfigState.__post_init__",
            step="get_config_level_artifact_states",
            context=f"dataset={self.dataset},config={self.config}",
        ):
            self.artifact_state_by_step = {
                processing_step.name: ArtifactState(
                    processing_step=processing_step,
                    dataset=self.dataset,
                    revision=self.revision,
                    config=self.config,
                    split=None,
                    error_codes_to_retry=self.error_codes_to_retry,
                    pending_jobs_df=self.pending_jobs_df[
                        (self.pending_jobs_df["split"].isnull())
                        & (self.pending_jobs_df["type"] == processing_step.job_type)
                    ],
                    cache_entries_df=self.cache_entries_df[
                        self.cache_entries_df["kind"] == processing_step.cache_kind
                    ],
                )
                for processing_step in self.processing_graph.get_input_type_processing_steps(input_type="config")
            }

        with StepProfiler(
            method="ConfigState.__post_init__",
            step="get_split_names",
            context=f"dataset={self.dataset},config={self.config}",
        ):
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
                )  # Note that we use the cached content even the revision is different (ie. maybe obsolete)
            except Exception:
                self.split_names = []

        with StepProfiler(
            method="ConfigState.__post_init__",
            step="get_split_states",
            context=f"dataset={self.dataset},config={self.config}",
        ):
            self.split_states = [
                SplitState(
                    self.dataset,
                    self.revision,
                    self.config,
                    split_name,
                    processing_graph=self.processing_graph,
                    error_codes_to_retry=self.error_codes_to_retry,
                    pending_jobs_df=self.pending_jobs_df[self.pending_jobs_df["split"] == split_name],
                    cache_entries_df=self.cache_entries_df[self.cache_entries_df["split"] == split_name],
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
    artifact_states_by_id: Dict[str, ArtifactState] = field(default_factory=dict)

    in_process: Dict[str, ArtifactState] = field(init=False)

    def __post_init__(self) -> None:
        self.in_process = {
            artifact_state.id: artifact_state
            for artifact_state in self.artifact_states_by_id.values()
            if artifact_state.job_state.is_in_process
        }

    def as_response(self) -> Dict[str, List[str]]:
        return {"in_process": sorted(self.in_process.keys())}


@dataclass
class Task(ABC):
    id: str = field(init=False)

    @abstractmethod
    def run(self) -> None:
        pass


@dataclass
class ArtifactTask(Task):
    artifact_state: ArtifactState


@dataclass
class CreateJobTask(ArtifactTask):
    priority: Priority

    def __post_init__(self) -> None:
        self.id = f"CreateJob,{self.artifact_state.id}"

    def run(self) -> None:
        with StepProfiler(
            method="CreateJobTask.run",
            step="all",
            context="creates 1 job",
        ):
            Queue().upsert_job(
                job_type=self.artifact_state.processing_step.job_type,
                dataset=self.artifact_state.dataset,
                revision=self.artifact_state.revision,
                config=self.artifact_state.config,
                split=self.artifact_state.split,
                priority=self.priority,
            )


@dataclass
class DeleteJobTask(ArtifactTask):
    def __post_init__(self) -> None:
        self.id = f"DeleteJob,{self.artifact_state.id}"

    def run(self) -> None:
        with StepProfiler(
            method="DeleteJobTask.run",
            step="all",
            context="deletes 1 job",
        ):
            Queue().cancel_jobs(
                job_type=self.artifact_state.processing_step.job_type,
                dataset=self.artifact_state.dataset,
                config=self.artifact_state.config,
                split=self.artifact_state.split,
                statuses_to_cancel=[Status.WAITING, Status.STARTED],
            )


@dataclass
class DeleteJobsTask(Task):
    jobs_df: pd.DataFrame

    def __post_init__(self) -> None:
        # for debug and testing
        artifact_ids = [
            Artifact.get_id(
                dataset=row["dataset"],
                revision=row["revision"],
                config=row["config"],
                split=row["split"],
                processing_step_name=row["type"],
            )
            for _, row in self.jobs_df.iterrows()
        ]
        self.id = f"DeleteJobs,{','.join(sorted(artifact_ids))}"

    def run(self) -> None:
        with StepProfiler(
            method="DeleteJobsTask.run",
            step="all",
            context=f"num_jobs_to_delete={len(self.jobs_df)}",
        ):
            if Queue().cancel_jobs_by_job_id(job_ids=self.jobs_df["job_id"].tolist()) != len(self.jobs_df):
                raise ValueError(
                    f"Something went wrong when cancelling jobs: {len(self.jobs_df)} jobs were supposed to be"
                    f" cancelled, but {len(self.jobs_df)} were cancelled."
                )


SupportedTask = Union[CreateJobTask, DeleteJobTask, DeleteJobsTask]


@dataclass
class Plan:
    tasks: List[SupportedTask] = field(default_factory=list)

    def add(self, task: SupportedTask) -> None:
        self.tasks.append(task)

    def run(self) -> int:
        """Run all the tasks in the plan.

        Returns:
            The number of tasks that were run.
        """
        for idx, task in enumerate(self.tasks):
            logging.debug(f"Running task [{idx} : {len(self.tasks)}]: {task.id}")
            task.run()
        return len(self.tasks)

    def as_response(self) -> List[str]:
        return sorted(task.id for task in self.tasks)


@dataclass
class DatasetState:
    """The state of a dataset."""

    dataset: str
    processing_graph: ProcessingGraph
    revision: str
    error_codes_to_retry: Optional[List[str]] = None
    priority: Priority = Priority.LOW

    pending_jobs_df: pd.DataFrame = field(init=False)
    cache_entries_df: pd.DataFrame = field(init=False)
    config_names: List[str] = field(init=False)
    config_states: List[ConfigState] = field(init=False)
    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)
    cache_status: CacheStatus = field(init=False)
    queue_status: QueueStatus = field(init=False)
    plan: Plan = field(init=False)
    should_be_backfilled: bool = field(init=False)

    def __post_init__(self) -> None:
        with StepProfiler(
            method="DatasetState.__post_init__",
            step="all",
            context=f"dataset={self.dataset}",
        ):
            with StepProfiler(
                method="DatasetState.__post_init__",
                step="get_pending_jobs_df",
                context=f"dataset={self.dataset}",
            ):
                self.pending_jobs_df = Queue().get_pending_jobs_df(dataset=self.dataset, revision=self.revision)
                self.pending_jobs_df = self.pending_jobs_df[
                    (self.pending_jobs_df["dataset"] == self.dataset)
                    & (self.pending_jobs_df["revision"] == self.revision)
                ]
                # ^ safety check
            with StepProfiler(
                method="DatasetState.__post_init__", step="get_cache_entries_df", context=f"dataset={self.dataset}"
            ):
                self.cache_entries_df = get_cache_entries_df(dataset=self.dataset)
                self.cache_entries_df = self.cache_entries_df[self.cache_entries_df["dataset"] == self.dataset]
                # ^ safety check

            with StepProfiler(
                method="DatasetState.__post_init__",
                step="get_dataset_level_artifact_states",
                context=f"dataset={self.dataset}",
            ):
                self.artifact_state_by_step = {
                    processing_step.name: ArtifactState(
                        processing_step=processing_step,
                        dataset=self.dataset,
                        revision=self.revision,
                        config=None,
                        split=None,
                        error_codes_to_retry=self.error_codes_to_retry,
                        pending_jobs_df=self.pending_jobs_df[
                            (self.pending_jobs_df["config"].isnull())
                            & (self.pending_jobs_df["split"].isnull())
                            & (self.pending_jobs_df["type"] == processing_step.job_type)
                        ],
                        cache_entries_df=self.cache_entries_df[
                            (self.cache_entries_df["kind"] == processing_step.cache_kind)
                            & (self.cache_entries_df["config"].isnull())
                            & (self.cache_entries_df["split"].isnull())
                        ],
                    )
                    for processing_step in self.processing_graph.get_input_type_processing_steps(input_type="dataset")
                }
            with StepProfiler(
                method="DatasetState.__post_init__",
                step="get_config_names",
                context=f"dataset={self.dataset}",
            ):
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
                    )  # Note that we use the cached content even the revision is different (ie. maybe obsolete)
                except Exception:
                    self.config_names = []
            with StepProfiler(
                method="DatasetState.__post_init__",
                step="get_config_states",
                context=f"dataset={self.dataset}",
            ):
                self.config_states = [
                    ConfigState(
                        dataset=self.dataset,
                        revision=self.revision,
                        config=config_name,
                        processing_graph=self.processing_graph,
                        error_codes_to_retry=self.error_codes_to_retry,
                        pending_jobs_df=self.pending_jobs_df[self.pending_jobs_df["config"] == config_name],
                        cache_entries_df=self.cache_entries_df[self.cache_entries_df["config"] == config_name],
                    )
                    for config_name in self.config_names
                ]
            with StepProfiler(
                method="DatasetState.__post_init__",
                step="_get_cache_status",
                context=f"dataset={self.dataset}",
            ):
                self.cache_status = self._get_cache_status()
            with StepProfiler(
                method="DatasetState.__post_init__",
                step="_get_queue_status",
                context=f"dataset={self.dataset}",
            ):
                self.queue_status = self._get_queue_status()
            with StepProfiler(
                method="DatasetState.__post_init__",
                step="_create_plan",
                context=f"dataset={self.dataset}",
            ):
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
        # remaining_in_process_artifact_state_ids = list(self.queue_status.in_process.keys())
        pending_jobs_to_delete_df = self.pending_jobs_df.copy()
        artifact_states = (
            list(self.cache_status.cache_is_empty.values())
            + list(self.cache_status.cache_is_error_to_retry.values())
            + list(self.cache_status.cache_is_outdated_by_parent.values())
            + list(self.cache_status.cache_is_job_runner_obsolete.values())
            + list(self.cache_status.cache_has_different_git_revision.values())
        )
        for artifact_state in artifact_states:
            valid_pending_jobs_df = artifact_state.job_state.valid_pending_jobs_df
            if valid_pending_jobs_df.empty:
                plan.add(CreateJobTask(artifact_state=artifact_state, priority=self.priority))
            else:
                pending_jobs_to_delete_df.drop(valid_pending_jobs_df.index, inplace=True)
        if not pending_jobs_to_delete_df.empty:
            plan.add(DeleteJobsTask(jobs_df=pending_jobs_to_delete_df))
        return plan

    def backfill(self) -> int:
        """Backfill the cache.

        Returns:
            The number of jobs created.
        """
        with StepProfiler(
            method="DatasetState.backfill",
            step="run",
            context=f"dataset={self.dataset}",
        ):
            logging.info(f"Backfilling {self.dataset}")
            return self.plan.run()

    def as_response(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "cache_status": self.cache_status.as_response(),
            "queue_status": self.queue_status.as_response(),
            "plan": self.plan.as_response(),
        }
