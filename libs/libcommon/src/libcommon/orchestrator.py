# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Union

import pandas as pd

from libcommon.processing_graph import (
    ProcessingGraph,
    ProcessingStep,
    ProcessingStepDoesNotExist,
)
from libcommon.prometheus import StepProfiler
from libcommon.queue import Queue
from libcommon.simple_cache import (
    fetch_names,
    get_cache_entries_df,
    has_some_cache,
    upsert_response_params,
)
from libcommon.state import ArtifactState, DatasetState, FirstStepsDatasetState
from libcommon.utils import JobInfo, JobResult, Priority

# TODO: clean dangling cache entries


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
    tasks: List[SupportedTask] = field(init=False)

    def __post_init__(self) -> None:
        self.tasks = []

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


@dataclass
class AfterJobPlan(Plan):
    """
    Plan to create jobs after a processing step has finished.

    Args:
        job_info (JobInfo): The job info.
        processing_graph (ProcessingGraph): The processing graph.
    """

    job_info: JobInfo
    processing_graph: ProcessingGraph

    dataset: str = field(init=False)
    config: Optional[str] = field(init=False)
    split: Optional[str] = field(init=False)
    revision: str = field(init=False)
    priority: Priority = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
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

        # Better keep this order: delete, then create
        # Note that all the pending jobs for other revisions will be deleted
        if not self.pending_jobs_df.empty:
            self.add_task(DeleteJobsTask(jobs_df=self.pending_jobs_df))
        if self.job_infos_to_create:
            self.add_task(CreateJobsTask(job_infos=self.job_infos_to_create))

    def update(
        self,
        next_processing_step: ProcessingStep,
        config: Optional[str],
        split: Optional[str],
    ) -> None:
        # ignore unrelated jobs
        config_mask = (
            self.pending_jobs_df["config"].isnull() if config is None else self.pending_jobs_df["config"] == config
        )
        split_mask = (
            self.pending_jobs_df["split"].isnull() if split is None else self.pending_jobs_df["split"] == split
        )

        unrelated_jobs_mask = (self.pending_jobs_df["type"] == next_processing_step.job_type) & (
            (self.pending_jobs_df["dataset"] != self.dataset) | (~config_mask) | (~split_mask)
        )
        self.pending_jobs_df = self.pending_jobs_df[~unrelated_jobs_mask]

        jobs_mask = (
            (self.pending_jobs_df["type"] == next_processing_step.job_type)
            & (self.pending_jobs_df["dataset"] == self.dataset)
            & (config_mask)
            & (split_mask)
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
class DatasetBackfillPlan(Plan):
    """
    Plan to backfill a dataset for a given revision.

    The plan is composed of tasks to delete and create jobs.

    Args:
        dataset: dataset name
        processing_graph: processing graph
        revision: revision to backfill
        error_codes_to_retry: list of error codes to retry
        priority: priority of the jobs to create
        only_first_processing_steps: if True, only the first processing steps are backfilled
    """

    dataset: str
    processing_graph: ProcessingGraph
    revision: str
    error_codes_to_retry: Optional[List[str]] = None
    priority: Priority = Priority.LOW
    only_first_processing_steps: bool = False

    pending_jobs_df: pd.DataFrame = field(init=False)
    cache_entries_df: pd.DataFrame = field(init=False)
    dataset_state: DatasetState = field(init=False)
    cache_status: CacheStatus = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()
        with StepProfiler(
            method="DatasetBackfillPlan.__post_init__",
            step="all",
            context=f"dataset={self.dataset}",
        ):
            with StepProfiler(
                method="DatasetBackfillPlan.__post_init__",
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
                method="DatasetBackfillPlan.__post_init__",
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
                method="DatasetBackfillPlan.__post_init__",
                step="get_dataset_state",
                context=f"dataset={self.dataset}",
            ):
                self.dataset_state = (
                    FirstStepsDatasetState(
                        dataset=self.dataset,
                        processing_graph=self.processing_graph,
                        revision=self.revision,
                        pending_jobs_df=self.pending_jobs_df,
                        cache_entries_df=self.cache_entries_df,
                        error_codes_to_retry=self.error_codes_to_retry,
                    )
                    if self.only_first_processing_steps
                    else DatasetState(
                        dataset=self.dataset,
                        processing_graph=self.processing_graph,
                        revision=self.revision,
                        pending_jobs_df=self.pending_jobs_df,
                        cache_entries_df=self.cache_entries_df,
                        error_codes_to_retry=self.error_codes_to_retry,
                    )
                )
            with StepProfiler(
                method="DatasetBackfillPlan.__post_init__",
                step="_get_cache_status",
                context=f"dataset={self.dataset}",
            ):
                self.cache_status = self._get_cache_status()
            with StepProfiler(
                method="DatasetBackfillPlan.__post_init__",
                step="_create_plan",
                context=f"dataset={self.dataset}",
            ):
                self._create_plan()

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
            artifact_states = [self.dataset_state.artifact_state_by_step[processing_step.name]]
        elif processing_step.input_type == "config":
            if config is None:
                artifact_states = [
                    config_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.dataset_state.config_states
                ]
            else:
                artifact_states = [
                    config_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.dataset_state.config_states
                    if config_state.config == config
                ]
        elif processing_step.input_type == "split":
            if config is None:
                artifact_states = [
                    split_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.dataset_state.config_states
                    for split_state in config_state.split_states
                ]
            elif split is None:
                artifact_states = [
                    split_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.dataset_state.config_states
                    if config_state.config == config
                    for split_state in config_state.split_states
                ]
            else:
                artifact_states = [
                    split_state.artifact_state_by_step[processing_step.name]
                    for config_state in self.dataset_state.config_states
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
                if artifact_state.cache_state.is_job_runner_obsolete():
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

    def _create_plan(self) -> None:
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
            self.add_task(DeleteJobsTask(jobs_df=pending_jobs_to_delete_df))
        if job_infos_to_create:
            self.add_task(CreateJobsTask(job_infos=job_infos_to_create))


@dataclass
class DatasetOrchestrator:
    dataset: str
    processing_graph: ProcessingGraph

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
            method="DatasetOrchestrator.set_revision",
            step="all",
            context=f"dataset={self.dataset}",
        ):
            logging.info(f"Analyzing {self.dataset}")
            with StepProfiler(
                method="DatasetOrchestrator.set_revision",
                step="plan",
                context=f"dataset={self.dataset}",
            ):
                plan = DatasetBackfillPlan(
                    dataset=self.dataset,
                    revision=revision,
                    priority=priority,
                    processing_graph=self.processing_graph,
                    error_codes_to_retry=error_codes_to_retry,
                    only_first_processing_steps=True,
                )
            logging.info(f"Setting new revision to {self.dataset}")
            with StepProfiler(
                method="DatasetOrchestrator.set_revision",
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
        # finish the job
        Queue().finish_job(job_id=job_info["job_id"], is_success=job_result["is_success"])
        logging.debug("the job has been finished.")
        # trigger the next steps
        plan = AfterJobPlan(job_info=job_info, processing_graph=self.processing_graph)
        plan.run()
        logging.debug("jobs have been created for the next steps.")

    def has_some_cache(self) -> bool:
        """
        Check if the cache has some entries for the dataset.

        Returns:
            bool: True if the cache has some entries for the dataset, False otherwise.
        """
        return has_some_cache(dataset=self.dataset)

    def has_pending_ancestor_jobs(self, processing_step_names: List[str]) -> bool:
        """
        Check if the processing steps, or one of their ancestors, have a pending job, ie. if artifacts could exist
          in the cache in the future. This method is used when a cache entry is missing in the API,
          to return a:
           - 404 error, saying that the artifact does not exist,
           - or a 500 error, saying that the artifact could be available soon (retry).

        It is implemented by checking if a job exists for the artifacts or one of their ancestors.

        Note that, if dataset-config-names' job is pending, we cannot know if the config is valid or not, so we
            consider that the artifact could exist.

        Args:
            processing_step_names (List[str]): The processing step names (artifacts) to check.

        Returns:
            bool: True if any of the artifact could exist, False otherwise.

        Raises:
            ValueError: If any of the processing step does not exist.
        """
        job_types: Set[str] = set()
        for processing_step_name in processing_step_names:
            try:
                processing_step = self.processing_graph.get_processing_step(processing_step_name)
            except ProcessingStepDoesNotExist as e:
                raise ValueError(f"Processing step {processing_step_name} does not exist") from e
            ancestors = self.processing_graph.get_ancestors(processing_step_name)
            job_types.add(processing_step.job_type)
            job_types.update(ancestor.job_type for ancestor in ancestors)
        # check if a pending job exists for the artifact or one of its ancestors
        # note that we cannot know if the ancestor is really for the artifact (ie: ancestor is for config1,
        # while we look for config2,split1). Looking in this detail would be too complex, this approximation
        # is good enough.
        return Queue().has_pending_jobs(dataset=self.dataset, job_types=list(job_types))

    def backfill(self, revision: str, priority: Priority, error_codes_to_retry: Optional[List[str]] = None) -> int:
        """
        Backfill the cache for a given revision.

        Args:
            revision (str): The revision.
            priority (Priority): The priority of the jobs.
            error_codes_to_retry (Optional[List[str]]): The error codes for which the jobs should be retried.

        Returns:
            int: The number of jobs created.
        """
        with StepProfiler(
            method="DatasetOrchestrator.backfill",
            step="all",
            context=f"dataset={self.dataset}",
        ):
            logging.info(f"Analyzing {self.dataset}")
            with StepProfiler(
                method="DatasetOrchestrator.backfill",
                step="plan",
                context=f"dataset={self.dataset}",
            ):
                plan = DatasetBackfillPlan(
                    dataset=self.dataset,
                    revision=revision,
                    priority=priority,
                    processing_graph=self.processing_graph,
                    error_codes_to_retry=error_codes_to_retry,
                    only_first_processing_steps=False,
                )
            logging.info(f"Analyzing {self.dataset}")
            with StepProfiler(
                method="DatasetOrchestrator.backfill",
                step="run",
                context=f"dataset={self.dataset}",
            ):
                return plan.run()
