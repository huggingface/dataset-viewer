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

    We don't delete everythig, then create everything, because the /metrics endpoint could be called at the same time,
    and the metrics would be inconsistent.
    """
    logging.info("updating cache metrics")
    new_count_entries = get_responses_count_by_kind_status_and_error_code()
    print(f"{new_count_entries=}")
    new_total_by_id = {
        (metric["kind"], metric["http_status"], metric["error_code"]): metric["count"] for metric in new_count_entries
    }
    print(f"{new_total_by_id=}")
    new_ids = set(new_total_by_id.keys())
    print(f"{new_ids=}")
    old_ids = set(
        (metric.kind, metric.http_status, metric.error_code) for metric in CacheTotalMetricDocument.objects()
    )
    print(f"{old_ids=}")
    to_delete = old_ids - new_ids
    print(f"{to_delete=}")

    for kind, http_status, error_code in to_delete:
        CacheTotalMetricDocument.objects(kind=kind, http_status=http_status, error_code=error_code).delete()
        print(f"{kind=} {http_status=} {error_code=} has been deleted")

    for (kind, http_status, error_code), total in new_total_by_id.items():
        CacheTotalMetricDocument.objects(kind=kind, http_status=http_status, error_code=error_code).upsert_one(
            total=total
        )
        print(f"{kind=} {http_status=} {error_code=}: {total=} has been inserted")

    logging.info("cache metrics have been updating")
