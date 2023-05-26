# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any, Dict, List, Mapping, Optional, Set, TypedDict, Union

import pandas as pd

from libcommon.processing_graph import ProcessingGraph, ProcessingStep, ProcessingStepDoesNotExist
from libcommon.prometheus import StepProfiler
from libcommon.queue import Queue
from libcommon.simple_cache import (
    CacheEntryMetadata,
    get_best_response,
    get_cache_entries_df,
    get_revision,
    upsert_response_params,
)
from libcommon.utils import JobInfo, Priority, inputs_to_string


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
    in_process: Set[str] = field(default_factory=set)

    def as_response(self) -> Dict[str, List[str]]:
        return {"in_process": sorted(self.in_process)}


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
class CreateJobsTask(Task):
    job_infos: List[JobInfo] = field(default_factory=list)

    def __post_init__(self) -> None:
        # for debug and testing
        self.id = f"CreateJobs,{len(self.job_infos)}"

    def run(self) -> None:
        with StepProfiler(
            method="CreateJobsTask.run",
            step="all",
            context=f"num_jobs_to_create={len(self.job_infos)}",
        ):
            created_jobs_count = Queue().create_jobs(job_infos=self.job_infos)
            if created_jobs_count != len(self.job_infos):
                raise ValueError(
                    f"Something went wrong when creating jobs: {len(self.job_infos)} jobs were supposed to be"
                    f" created, but {created_jobs_count} were created."
                )


@dataclass
class DeleteJobsTask(Task):
    jobs_df: pd.DataFrame

    def __post_init__(self) -> None:
        # for debug and testing
        self.id = f"DeleteJobs,{len(self.jobs_df)}"

    def run(self) -> None:
        with StepProfiler(
            method="DeleteJobsTask.run",
            step="all",
            context=f"num_jobs_to_delete={len(self.jobs_df)}",
        ):
            cancelled_jobs_count = Queue().cancel_jobs_by_job_id(job_ids=self.jobs_df["job_id"].tolist())
            if cancelled_jobs_count != len(self.jobs_df):
                raise ValueError(
                    f"Something went wrong when cancelling jobs: {len(self.jobs_df)} jobs were supposed to be"
                    f" cancelled, but {cancelled_jobs_count} were cancelled."
                )


SupportedTask = Union[CreateJobsTask, DeleteJobsTask]


@dataclass
class Plan:
    tasks: List[SupportedTask] = field(default_factory=list)

    def add_task(self, task: SupportedTask) -> None:
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


class JobResult(TypedDict):
    job_info: JobInfo
    job_runner_version: int
    is_success: bool
    output: Optional[JobOutput]


class JobOutput(TypedDict):
    content: Mapping[str, Any]
    http_status: HTTPStatus
    error_code: Optional[str]
    details: Optional[Mapping[str, Any]]
    progress: Optional[float]


@dataclass
class Orchestrator:
    dataset: str
    processing_graph: ProcessingGraph

    def get_revision(self) -> Optional[str]:
        """
        Get the current revision of the dataset.

        It's stored in the cache entries of the dataset itself, and we get it from the first processing step.

        Returns:
            The current revision of the dataset. None if not found.

        Raises:
            ValueError: If the first processing step is not a dataset step, or if the processing graph has no first
              step.
        """
        first_processing_steps = self.processing_graph.get_first_processing_steps()
        if len(first_processing_steps) < 1:
            raise ValueError("Processing graph has no first step")
        first_processing_step = first_processing_steps[0]
        if first_processing_step.input_type != "dataset":
            raise ValueError("First processing step is not a dataset step")
        return get_revision(kind=first_processing_step.cache_kind, dataset=self.dataset, config=None, split=None)

    def set_revision(self, revision: str, priority: Priority, error_codes_to_retry: List[str]) -> None:
        """
        Set the current revision of the dataset.

        If the revision is already set to the same value, this is a no-op. Else: one job is created for every first
          step.

        Args:
            revision (str): The new revision of the dataset.
            priority (Priority): The priority of the jobs to create.
            error_codes_to_retry (List[str]): The error codes for which the jobs should be retried.

        Returns:
            None

        Raises:
            ValueError: If the first processing steps are not dataset steps, or if the processing graph has no first
              step.
        """
        first_processing_steps = self.processing_graph.get_first_processing_steps()
        if len(first_processing_steps) < 1:
            raise ValueError("Processing graph has no first step")
        if any(first_processing_step.input_type != "dataset" for first_processing_step in first_processing_steps):
            raise ValueError("One of the first processing steps is not a dataset step")
        with StepProfiler(
            method="Orchestrator.set_revision",
            step="all",
            context=f"dataset={self.dataset}",
        ):
            logging.info(f"Analyzing {self.dataset}")
            with StepProfiler(
                method="Orchestrator.set_revision",
                step="plan",
                context=f"dataset={self.dataset}",
            ):
                plan = DatasetState(
                    dataset=self.dataset,
                    revision=revision,
                    priority=priority,
                    processing_graph=self.processing_graph,
                    error_codes_to_retry=error_codes_to_retry,
                    only_first_processing_steps=True,
                ).plan
            logging.info(f"Setting new revision to {self.dataset}")
            with StepProfiler(
                method="Orchestrator.set_revision",
                step="run",
                context=f"dataset={self.dataset}",
            ):
                plan.run()

    def finish_job(self, job_result: JobResult) -> None:
        """
        Finish a job.

        It will finish the job, store the result in the cache, and trigger the next steps.

        Args:
            job_result (JobResult): The result of the job.

        Returns:
            None

        Raises:
            ValueError: If the job is not found, or if the processing step is not found.
        """
        # check if the job is still in started status
        job_info = job_result["job_info"]
        if not Queue().is_job_started(job_id=job_info["job_id"]):
            logging.debug("the job was cancelled, don't update the cache")
            return
        # if the job could not provide an output, finish it and return
        if not job_result["output"]:
            Queue().finish_job(job_id=job_info["job_id"], is_success=False)
            logging.debug("the job raised an exception, don't update the cache")
            return
        # update the cache
        output = job_result["output"]
        params = job_info["params"]
        try:
            processing_step = self.processing_graph.get_processing_step_by_job_type(job_info["type"])
        except ProcessingStepDoesNotExist as e:
            raise ValueError(f"Processing step for job type {job_info['type']} does not exist") from e
        upsert_response_params(
            # inputs
            kind=processing_step.cache_kind,
            job_params=params,
            job_runner_version=job_result["job_runner_version"],
            # output
            content=output["content"],
            http_status=output["http_status"],
            error_code=output["error_code"],
            details=output["details"],
            progress=output["progress"],
        )
        logging.debug("the job output has been written to the cache.")
        # trigger the next steps
        plan = ChildrenJobsCreation(job_info=job_info, processing_graph=self.processing_graph).plan
        plan.run()
        logging.debug("jobs have been created for the next steps.")
        # finish the job
        Queue().finish_job(job_id=job_info["job_id"], is_success=job_result["is_success"])
        logging.debug("the job has been finished.")

    def could_artifact_exist(self, processing_step_name: str, revision: str) -> bool:
        """
        Check if an artifact could exist in the cache. This method is used when a cache entry is missing in the API,
          to return a 404 error, saying that the artifact does not exist, or a 500 error, saying that the artifact
            should exist and will soon be available (retry).

        It is implemented by checking if a job exists for the artifact or one of its ancestors.

        Note that, if dataset-config-names' job is pending, we cannot know if the config is valid or not, so we
            consider that the artifact could exist.

        Args:
            processing_step_name (str): The processing step name.
            revision (str): The revision of the dataset.

        Returns:
            bool: True if the artifact could exist, False otherwise.

        Raises:
            ValueError: If the processing step does not exist.
        """
        try:
            processing_step = self.processing_graph.get_processing_step(processing_step_name)
        except ProcessingStepDoesNotExist as e:
            raise ValueError(f"Processing step {processing_step_name} does not exist") from e
        ancestors = self.processing_graph.get_ancestors(processing_step_name)
        processing_steps = [processing_step] + ancestors
        # check if a pending job exists for the artifact or one of its ancestors
        # note that we cannot know if the ancestor is really for the artifact (ie: ancestor is for config1,
        # while we look for config2,split1). Looking in this detail would be too complex, this approximation
        # is good enough.
        return Queue().has_pending_jobs_df(
            dataset=self.dataset,
            revision=revision,
            job_types=[processing_step.job_type for processing_step in processing_steps],
        )

    def backfill(self, revision: str, priority: Priority, error_codes_to_retry: Optional[List[str]] = None) -> None:
        """
        Backfill the cache for a given revision.

        Args:
            revision (str): The revision.
            priority (Priority): The priority of the jobs.
            error_codes_to_retry (Optional[List[str]]): The error codes for which the jobs should be retried.

        Returns:
            None
        """
        with StepProfiler(
            method="DatasetState.backfill",
            step="all",
            context=f"dataset={self.dataset}",
        ):
            logging.info(f"Analyzing {self.dataset}")
            with StepProfiler(
                method="DatasetState.backfill",
                step="plan",
                context=f"dataset={self.dataset}",
            ):
                plan = DatasetState(
                    dataset=self.dataset,
                    revision=revision,
                    priority=priority,
                    processing_graph=self.processing_graph,
                    error_codes_to_retry=error_codes_to_retry,
                ).plan
            logging.info(f"Analyzing {self.dataset}")
            with StepProfiler(
                method="DatasetState.backfill",
                step="run",
                context=f"dataset={self.dataset}",
            ):
                plan.run()


@dataclass
class ChildrenJobsCreation:
    """
    Create jobs for the children of a processing step.

    Args:
        job_info (JobInfo): The job info.
        processing_graph (ProcessingGraph): The processing graph.

    Returns:
        Plan: The plan.
    """

    job_info: JobInfo
    processing_graph: ProcessingGraph

    dataset: str = field(init=False)
    config: Optional[str] = field(init=False)
    split: Optional[str] = field(init=False)
    revision: str = field(init=False)
    priority: Priority = field(init=False)
    plan: Plan = field(init=False)

    def __post_init__(self) -> None:
        self.dataset = self.job_info["params"]["dataset"]
        self.revision = self.job_info["params"]["revision"]
        self.priority = self.job_info["priority"]

        config = self.job_info["params"]["config"]
        split = self.job_info["params"]["split"]
        job_type = self.job_info["type"]
        try:
            processing_step = self.processing_graph.get_processing_step_by_job_type(job_type)
            next_processing_steps = self.processing_graph.get_children(processing_step.name)
        except ProcessingStepDoesNotExist as e:
            raise ValueError(f"Processing step with job type: {job_type} does not exist") from e

        # get the list of pending jobs for the children
        # note that it can contain a lot of unrelated jobs, we will clean after
        self.pending_jobs_df = Queue().get_pending_jobs_df(
            dataset=self.dataset,
            job_types=[next_processing_step.job_type for next_processing_step in next_processing_steps],
        )

        self.job_infos_to_create: List[JobInfo] = []
        config_names: Optional[List[str]] = None
        split_names: Optional[List[str]] = None

        # filter to only get the jobs that are not already in the queue
        for next_processing_step in next_processing_steps:
            if processing_step.input_type == next_processing_step.input_type:
                # same level, one job is expected
                # D -> D, C -> C, S -> S
                self.update(next_processing_step, config, split)
            elif processing_step.input_type in ["config", "split"] and next_processing_step.input_type == "dataset":
                # going to upper level (fan-in), one job is expected
                # S -> D, C -> D
                self.update(next_processing_step, None, None)
            elif processing_step.input_type == "split" and next_processing_step.input_type == "config":
                # going to upper level (fan-in), one job is expected
                # S -> C
                self.update(next_processing_step, config, None)
            elif processing_step.input_type == "dataset" and next_processing_step.input_type == "config":
                # going to lower level (fan-out), one job is expected per config, we need the list of configs
                # D -> C
                if config_names is None:
                    config_names = fetch_names(
                        dataset=self.dataset,
                        config=None,
                        cache_kinds=[
                            processing_step.cache_kind
                            for processing_step in self.processing_graph.get_dataset_config_names_processing_steps()
                        ],
                        names_field="config_names",
                        name_field="config",
                    )  # Note that we use the cached content even the revision is different (ie. maybe obsolete)
                for config_name in config_names:
                    self.update(next_processing_step, config_name, None)
            elif processing_step.input_type == "config" and next_processing_step.input_type == "split":
                # going to lower level (fan-out), one job is expected per split, we need the list of splits
                # C -> S
                if split_names is None:
                    split_names = fetch_names(
                        dataset=self.dataset,
                        config=config,
                        cache_kinds=[
                            processing_step.cache_kind
                            for processing_step in self.processing_graph.get_config_split_names_processing_steps()
                        ],
                        names_field="split_names",
                        name_field="split",
                    )  # Note that we use the cached content even the revision is different (ie. maybe obsolete)
                for split_name in split_names:
                    self.update(next_processing_step, config, split_name)
            else:
                raise NotImplementedError(
                    f"Unsupported input types: {processing_step.input_type} -> {next_processing_step.input_type}"
                )
                # we don't support fan-out dataset-level to split-level (no need for now)

        self.plan = Plan()
        # Better keep this order: delete, then create
        # Note that all the pending jobs for other revisions will be deleted
        if not self.pending_jobs_df.empty:
            self.plan.add_task(DeleteJobsTask(jobs_df=self.pending_jobs_df))
        if self.job_infos_to_create:
            self.plan.add_task(CreateJobsTask(job_infos=self.job_infos_to_create))

    def update(
        self,
        next_processing_step: ProcessingStep,
        config: Optional[str],
        split: Optional[str],
    ) -> None:
        # ignore unrelated jobs
        unrelated_jobs_mask = (self.pending_jobs_df["job_type"] == next_processing_step.job_type) & (
            (self.pending_jobs_df["params.dataset"] != self.dataset)
            | (self.pending_jobs_df["params.config"] != config)
            | (self.pending_jobs_df["params.split"] != split)
        )
        self.pending_jobs_df = self.pending_jobs_df[~unrelated_jobs_mask]

        jobs_mask = (
            (self.pending_jobs_df["job_type"] == next_processing_step.job_type)
            & (self.pending_jobs_df["params.dataset"] == self.dataset)
            & (self.pending_jobs_df["params.config"] == config)
            & (self.pending_jobs_df["params.split"] == split)
        )
        ok_jobs_mask = jobs_mask & (self.pending_jobs_df["revision"] == self.revision)
        if ok_jobs_mask.any():
            # remove the first ok job for the list, and keep the others to delete them later
            self.pending_jobs_df.drop(ok_jobs_mask.idxmax(), inplace=True)
        else:
            # no pending job for the current processing step
            self.job_infos_to_create.append(
                {
                    "job_id": "not used",  # TODO: remove this field
                    "type": next_processing_step.job_type,
                    "params": {
                        "dataset": self.dataset,
                        "config": config,
                        "split": split,
                        "revision": self.revision,
                    },
                    "priority": self.priority,
                }
            )


@dataclass
class DatasetState:
    """Dataset state."""

    dataset: str
    processing_graph: ProcessingGraph
    revision: str
    error_codes_to_retry: Optional[List[str]] = None
    priority: Priority = Priority.LOW
    only_first_processing_steps: bool = False

    pending_jobs_df: pd.DataFrame = field(init=False)
    cache_entries_df: pd.DataFrame = field(init=False)
    config_names: List[str] = field(init=False)
    config_states: List[ConfigState] = field(init=False)
    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)
    cache_status: CacheStatus = field(init=False)
    plan: Plan = field(init=False)

    def __post_init__(self) -> None:
        with StepProfiler(
            method="BackfillJobsCreation.__post_init__",
            step="all",
            context=f"dataset={self.dataset}",
        ):
            with StepProfiler(
                method="BackfillJobsCreation.__post_init__",
                step="get_pending_jobs_df",
                context=f"dataset={self.dataset}",
            ):
                job_types = (
                    [
                        processing_step.job_type
                        for processing_step in self.processing_graph.get_first_processing_steps()
                    ]
                    if self.only_first_processing_steps
                    else None
                )
                self.pending_jobs_df = Queue().get_pending_jobs_df(
                    dataset=self.dataset,
                    job_types=job_types,
                )
            with StepProfiler(
                method="BackfillJobsCreation.__post_init__",
                step="get_cache_entries_df",
                context=f"dataset={self.dataset}",
            ):
                cache_kinds = (
                    [
                        processing_step.cache_kind
                        for processing_step in self.processing_graph.get_first_processing_steps()
                    ]
                    if self.only_first_processing_steps
                    else None
                )
                self.cache_entries_df = get_cache_entries_df(
                    dataset=self.dataset,
                    cache_kinds=cache_kinds,
                )

            with StepProfiler(
                method="BackfillJobsCreation.__post_init__",
                step="get_dataset_level_artifact_states",
                context=f"dataset={self.dataset}",
            ):
                dataset_level_processing_steps = (
                    self.processing_graph.get_first_processing_steps()
                    if self.only_first_processing_steps
                    else self.processing_graph.get_input_type_processing_steps(input_type="dataset")
                )
                self.artifact_state_by_step = {
                    processing_step.name: ArtifactState(
                        processing_step=processing_step,
                        dataset=self.dataset,
                        revision=self.revision,
                        config=None,
                        split=None,
                        error_codes_to_retry=self.error_codes_to_retry,
                        pending_jobs_df=self.pending_jobs_df[
                            (self.pending_jobs_df["revision"] == self.revision)
                            & (self.pending_jobs_df["config"].isnull())
                            & (self.pending_jobs_df["split"].isnull())
                            & (self.pending_jobs_df["type"] == processing_step.job_type)
                        ],
                        cache_entries_df=self.cache_entries_df[
                            (self.cache_entries_df["kind"] == processing_step.cache_kind)
                            & (self.cache_entries_df["config"].isnull())
                            & (self.cache_entries_df["split"].isnull())
                        ],
                    )
                    for processing_step in dataset_level_processing_steps
                }
            if self.only_first_processing_steps:
                self.config_names = []
                self.config_states = []
            else:
                with StepProfiler(
                    method="BackfillJobsCreation.__post_init__",
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
                    method="BackfillJobsCreation.__post_init__",
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
                            pending_jobs_df=self.pending_jobs_df[
                                (self.pending_jobs_df["revision"] == self.revision)
                                & (self.pending_jobs_df["config"] == config_name)
                            ],
                            cache_entries_df=self.cache_entries_df[self.cache_entries_df["config"] == config_name],
                        )
                        for config_name in self.config_names
                    ]
            with StepProfiler(
                method="BackfillJobsCreation.__post_init__",
                step="_get_cache_status",
                context=f"dataset={self.dataset}",
            ):
                self.cache_status = self._get_cache_status()
            with StepProfiler(
                method="BackfillJobsCreation.__post_init__",
                step="_create_plan",
                context=f"dataset={self.dataset}",
            ):
                self.plan = self._create_plan()

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

        processing_steps = (
            self.processing_graph.get_first_processing_steps()
            if self.only_first_processing_steps
            else self.processing_graph.get_topologically_ordered_processing_steps()
        )
        for processing_step in processing_steps:
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

    def get_queue_status(self) -> QueueStatus:
        processing_steps = (
            self.processing_graph.get_first_processing_steps()
            if self.only_first_processing_steps
            else self.processing_graph.get_topologically_ordered_processing_steps()
        )
        return QueueStatus(
            in_process={
                artifact_state.id
                for processing_step in processing_steps
                for artifact_state in self._get_artifact_states_for_step(processing_step)
                if artifact_state.job_state.is_in_process
            }
        )

    def _create_plan(self) -> Plan:
        plan = Plan()
        pending_jobs_to_delete_df = self.pending_jobs_df.copy()
        job_infos_to_create: List[JobInfo] = []
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
                job_infos_to_create.append(
                    {
                        "job_id": "not used",
                        "type": artifact_state.processing_step.job_type,
                        "params": {
                            "dataset": self.dataset,
                            "revision": self.revision,
                            "config": artifact_state.config,
                            "split": artifact_state.split,
                        },
                        "priority": self.priority,
                    }
                )
            else:
                pending_jobs_to_delete_df.drop(valid_pending_jobs_df.index, inplace=True)
        # Better keep this order: delete, then create
        # Note that all the pending jobs for other revisions will be deleted
        if not pending_jobs_to_delete_df.empty:
            plan.add_task(DeleteJobsTask(jobs_df=pending_jobs_to_delete_df))
        if job_infos_to_create:
            plan.add_task(CreateJobsTask(job_infos=job_infos_to_create))
        return plan

    def as_response(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "revision": self.revision,
            "cache_status": self.cache_status.as_response(),
            "queue_status": self.get_queue_status().as_response(),
            "plan": self.plan.as_response(),
        }
