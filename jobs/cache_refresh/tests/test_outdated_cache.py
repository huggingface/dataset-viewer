# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import List

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingStep
from libcommon.queue import Queue
from libcommon.simple_cache import upsert_response

from cache_refresh.outdated_cache import refresh_cache


def test_refresh_cache(processing_steps: List[ProcessingStep], processing_graph_config: ProcessingGraphConfig) -> None:
    queue = Queue()
    assert not queue.is_job_in_process(job_type="/config-names", dataset="dataset")
    upsert_response(
        kind="/config-names", dataset="dataset", content={}, http_status=HTTPStatus.OK, worker_version="0.0.0"
    )
    refresh_cache(processing_steps=processing_steps, processing_graph_config=processing_graph_config)
    assert queue.is_job_in_process(job_type="/config-names", dataset="dataset")
    # TODO: Add more tests
