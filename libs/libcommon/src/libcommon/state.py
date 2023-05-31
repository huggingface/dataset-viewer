# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import pandas as pd

from libcommon.processing_graph import ProcessingGraph, ProcessingStep
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import CacheEntryMetadata, fetch_names
from libcommon.utils import inputs_to_string

# TODO: assets, cached_assets, parquet files


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
    job_runner_version: int
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

    def is_job_runner_obsolete(self) -> bool:
        if self.cache_entry_metadata is None:
            return False
        if self.cache_entry_metadata["job_runner_version"] is None:
            return True
        return self.cache_entry_metadata["job_runner_version"] < self.job_runner_version


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
            job_runner_version=self.processing_step.job_runner_version,
            error_codes_to_retry=self.error_codes_to_retry,
            cache_entries_df=self.cache_entries_df,
        )


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
class DatasetState:
    """The state of a dataset."""

    dataset: str
    revision: str
    processing_graph: ProcessingGraph
    pending_jobs_df: pd.DataFrame
    cache_entries_df: pd.DataFrame
    error_codes_to_retry: Optional[List[str]] = None

    config_names: List[str] = field(init=False)
    config_states: List[ConfigState] = field(init=False)
    artifact_state_by_step: Dict[str, ArtifactState] = field(init=False)

    def __post_init__(self) -> None:
        with StepProfiler(
            method="DatasetBackfillPlan.__post_init__",
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
                for processing_step in self.processing_graph.get_input_type_processing_steps(input_type="dataset")
            }

            with StepProfiler(
                method="DatasetBackfillPlan.__post_init__",
                step="get_config_names",
                context=f"dataset={self.dataset}",
            ):
                self.config_names = fetch_names(
                    dataset=self.dataset,
                    config=None,
                    cache_kinds=[
                        step.cache_kind for step in self.processing_graph.get_dataset_config_names_processing_steps()
                    ],
                    names_field="config_names",
                    name_field="config",
                )  # Note that we use the cached content even the revision is different (ie. maybe obsolete)

            with StepProfiler(
                method="DatasetBackfillPlan.__post_init__",
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


@dataclass
class FirstStepsDatasetState(DatasetState):
    """The state of the first dataset steps."""

    def __post_init__(self) -> None:
        with StepProfiler(
            method="DatasetBackfillPlan.__post_init__",
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
                for processing_step in self.processing_graph.get_first_processing_steps()
            }

            self.config_names = []
            self.config_states = []
