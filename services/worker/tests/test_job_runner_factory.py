# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import Optional

import pytest
from libcommon.processing_graph import ProcessingGraph
from libcommon.storage import StrPath
from libcommon.utils import JobInfo, Priority

from worker.config import AppConfig
from worker.job_runner_factory import JobRunnerFactory
from worker.resources import LibrariesResource


@pytest.fixture()
def processing_graph(app_config: AppConfig) -> ProcessingGraph:
    return ProcessingGraph(app_config.processing_graph.specification)


@pytest.mark.parametrize(
    "job_type,expected_job_runner",
    [
        ("/config-names", "ConfigNamesJobRunner"),
        ("split-first-rows-from-streaming", "SplitFirstRowsFromStreamingJobRunner"),
        ("config-parquet-and-info", "ConfigParquetAndInfoJobRunner"),
        ("config-parquet", "ConfigParquetJobRunner"),
        ("dataset-parquet", "DatasetParquetJobRunner"),
        ("config-info", "ConfigInfoJobRunner"),
        ("dataset-info", "DatasetInfoJobRunner"),
        ("config-size", "ConfigSizeJobRunner"),
        ("dataset-size", "DatasetSizeJobRunner"),
        ("/unknown", None),
    ],
)
def test_create_job_runner(
    app_config: AppConfig,
    processing_graph: ProcessingGraph,
    libraries_resource: LibrariesResource,
    assets_directory: StrPath,
    job_type: str,
    expected_job_runner: Optional[str],
) -> None:
    factory = JobRunnerFactory(
        app_config=app_config,
        processing_graph=processing_graph,
        hf_datasets_cache=libraries_resource.hf_datasets_cache,
        assets_directory=assets_directory,
    )
    job_info: JobInfo = {
        "type": job_type,
        "dataset": "dataset",
        "config": "config",
        "split": "split",
        "job_id": "job_id",
        "force": False,
        "priority": Priority.NORMAL,
    }
    if expected_job_runner is None:
        with pytest.raises(KeyError):
            factory.create_job_runner(job_info=job_info)
    else:
        job_runner = factory.create_job_runner(job_info=job_info)
        assert job_runner.__class__.__name__ == expected_job_runner
