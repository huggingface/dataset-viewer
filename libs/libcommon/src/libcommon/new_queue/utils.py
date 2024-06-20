# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.

from ..queue import JobDocument, JobTotalMetricDocument, WorkerSizeJobsCountDocument
from .lock import Lock


# only for the tests
def _clean_queue_database() -> None:
    """Delete all the jobs in the database"""
    JobDocument.drop_collection()  # type: ignore
    JobTotalMetricDocument.drop_collection()  # type: ignore
    WorkerSizeJobsCountDocument.drop_collection()  # type: ignore
    Lock.drop_collection()  # type: ignore
