# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import List

import pytest
from libcommon.config import ProcessingGraphConfig
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
    processing_steps: List[ProcessingStep],
    processing_graph_config: ProcessingGraphConfig,
    job_runner_version: int,
    upserts_job: bool,
) -> None:
    queue = Queue()
    assert not queue.is_job_in_process(job_type="/config-names", dataset="dataset")
    upsert_response(
        kind="/config-names", dataset="dataset", content={}, http_status=HTTPStatus.OK, job_runner_version=job_runner_version
    )
    refresh_cache(processing_steps=processing_steps, processing_graph_config=processing_graph_config)
    assert queue.is_job_in_process(job_type="/config-names", dataset="dataset") == upserts_job
