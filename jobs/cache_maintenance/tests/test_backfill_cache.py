# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from unittest.mock import patch

from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Queue, Status

from cache_maintenance.backfill import backfill_cache


def test_backfill_cache() -> None:
    some_datasets = ["dummy", "dummy_2", "dummy_3"]
    datasets_number = len(some_datasets)
    step_name = "dummy_step"
    with patch("libcommon.dataset.get_supported_datasets", return_value=some_datasets):
        init_processing_steps = [
            ProcessingStep(
                name=step_name,
                input_type="dataset",
                requires=None,
                required_by_dataset_viewer=False,
                parent=None,
                ancestors=[],
                children=[],
                job_runner_version=1,
            )
        ]
        backfill_cache(init_processing_steps=init_processing_steps, hf_endpoint="hf_endpoint", hf_token="hf_token")
    queue = Queue()
    assert queue.count_jobs(Status.WAITING, step_name) == datasets_number
    assert queue.count_jobs(Status.STARTED, step_name) == 0
