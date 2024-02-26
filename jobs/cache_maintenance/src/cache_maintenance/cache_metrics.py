# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import logging

from libcommon.simple_cache import (
    CacheTotalMetricDocument,
    get_responses_count_by_kind_status_and_error_code,
)


def collect_cache_metrics() -> None:
    """
    Collects cache metrics and updates the cache metrics in the database.

    The obsolete cache metrics are deleted, and the new ones are inserted or updated.

    We don't delete everything, then create everything, because the /metrics endpoint could be called at the same time,
    and the metrics would be inconsistent.
    """
    logging.info("updating cache metrics")
    new_metric_by_id = get_responses_count_by_kind_status_and_error_code()
    new_ids = set(new_metric_by_id.keys())
    old_ids = set(
        (metric.kind, metric.http_status, metric.error_code) for metric in CacheTotalMetricDocument.objects()
    )
    to_delete = old_ids - new_ids

    for kind, http_status, error_code in to_delete:
        CacheTotalMetricDocument.objects(kind=kind, http_status=http_status, error_code=error_code).delete()
        logging.info(f"{kind=} {http_status=} {error_code=} has been deleted")

    for (kind, http_status, error_code), total in new_metric_by_id.items():
        CacheTotalMetricDocument.objects(kind=kind, http_status=http_status, error_code=error_code).upsert_one(
            total=total
        )
        logging.info(f"{kind=} {http_status=} {error_code=}: {total=} has been inserted")

    logging.info("cache metrics have been updated")
