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
        kind = metric["kind"]
        http_status = metric["http_status"]
        error_code = metric["error_code"]
        new_total = metric["count"]

        query_set = CacheTotalMetricDocument.objects(kind=kind, http_status=http_status, error_code=error_code)
        current_metric = query_set.first()
        if current_metric is not None:
            current_total = current_metric.total
            logging.info(
                f"{kind=} {http_status=} {error_code=}  current_total={current_total} new_total="
                f"{new_total} difference={new_total-current_total}"
            )
        query_set.upsert_one(total=metric["count"])
    logging.info("metrics have been collected")
