# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.simple_cache import (
    CacheTotalMetricDocument,
    get_responses_count_by_kind_status_and_error_code,
)


def collect_cache_metrics() -> None:
    logging.info("collecting cache metrics")
    for metric in get_responses_count_by_kind_status_and_error_code():
        # TODO: Get difference and store it somewhere?
        # TODO: Maybe initially just log the difference
        CacheTotalMetricDocument.objects(
            kind=metric["kind"], http_status=metric["http_status"], error_code=metric["error_code"]
        ).upsert_one(total=metric["count"])
    logging.info("metrics have been collected")
