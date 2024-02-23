# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from unittest.mock import patch

import pytest
from libcommon.simple_cache import CacheTotalMetricDocument, CountEntry

from cache_maintenance.cache_metrics import collect_cache_metrics

KIND_A = "kindA"
STATUS_500 = 500
ERROR_A = "ErrorA"
COUNT = 1

COUNT_ENTRY = CountEntry(kind=KIND_A, http_status=STATUS_500, error_code=ERROR_A, count=COUNT)
COUNT_ENTRY_KIND = CountEntry(kind="kindB", http_status=STATUS_500, error_code=ERROR_A, count=COUNT)
COUNT_ENTRY_STATUS = CountEntry(kind=KIND_A, http_status=400, error_code=ERROR_A, count=COUNT)
COUNT_ENTRY_ERROR = CountEntry(kind=KIND_A, http_status=STATUS_500, error_code="ErrorB", count=COUNT)
COUNT_ENTRY_ERROR_NONE = CountEntry(kind=KIND_A, http_status=STATUS_500, error_code=None, count=COUNT)
COUNT_ENTRY_COUNT = CountEntry(kind=KIND_A, http_status=STATUS_500, error_code=ERROR_A, count=COUNT + 1)


@pytest.mark.parametrize(
    "metrics,new_count_entries",
    [
        ([], [COUNT_ENTRY]),
        ([COUNT_ENTRY], [COUNT_ENTRY]),
        ([COUNT_ENTRY_KIND], [COUNT_ENTRY]),
        ([COUNT_ENTRY_STATUS], [COUNT_ENTRY]),
        ([COUNT_ENTRY_ERROR], [COUNT_ENTRY]),
        ([COUNT_ENTRY_ERROR_NONE], [COUNT_ENTRY]),
        ([COUNT_ENTRY_COUNT], [COUNT_ENTRY]),
    ],
)
def test_collect_cache_metrics(metrics: list[CountEntry], new_count_entries: list[CountEntry]) -> None:
    for metric in metrics:
        CacheTotalMetricDocument(
            kind=metric["kind"],
            http_status=metric["http_status"],
            error_code=metric["error_code"],
            total=metric["count"],
        ).save()

    with patch(
        "cache_maintenance.cache_metrics.get_responses_count_by_kind_status_and_error_code",
        return_value=new_count_entries,
    ):
        collect_cache_metrics()

    assert [
        CountEntry(kind=metric.kind, http_status=metric.http_status, error_code=metric.error_code, count=metric.total)
        for metric in CacheTotalMetricDocument.objects()
    ] == new_count_entries
