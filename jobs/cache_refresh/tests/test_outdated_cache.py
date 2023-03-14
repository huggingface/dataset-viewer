# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus

import pytest
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Queue
from libcommon.simple_cache import upsert_response

from cache_refresh.outdated_cache import refresh_cache


@pytest.mark.parametrize(
    "job_runner_version, upserts_job",
    [
        (0, True),
        (1, False),
        (2, False),
    ],
)
def test_refresh_cache(
    job_runner_version: int,
    upserts_job: bool,
) -> None:
    processing_steps = [
        ProcessingStep(
            name="/config-names",
            input_type="dataset",
            requires=None,
            required_by_dataset_viewer=False,
            parent=None,
            ancestors=[],
            children=[],
            job_runner_version=1,
        )
    ]
    queue = Queue()
    assert not queue.is_job_in_process(job_type="/config-names", dataset="dataset")
    upsert_response(
        kind="/config-names",
        dataset="dataset",
        content={},
        http_status=HTTPStatus.OK,
        job_runner_version=job_runner_version,
    )
    refresh_cache(processing_steps=processing_steps)
    assert queue.is_job_in_process(job_type="/config-names", dataset="dataset") == upserts_job
