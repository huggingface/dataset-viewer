# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from unittest.mock import patch

import pytest
from libcommon.simple_cache import CacheTotalMetricDocument, EntriesTotalByKindStatusAndErrorCode

from cache_maintenance.cache_metrics import collect_cache_metrics

KIND_A = "kindA"
STATUS_500 = 500
ERROR_A = "ErrorA"
COUNT = 1

NEW_METRIC = {(KIND_A, STATUS_500, ERROR_A): COUNT}
OTHER_KIND = {("kindB", STATUS_500, ERROR_A): COUNT}
OTHER_STATUS = {(KIND_A, 400, ERROR_A): COUNT}
OTHER_ERROR = {(KIND_A, STATUS_500, "ErrorB"): COUNT}
NONE_ERROR = {(KIND_A, STATUS_500, None): COUNT}
OTHER_COUNT = {(KIND_A, STATUS_500, ERROR_A): COUNT + 1}


@pytest.mark.parametrize(
    "old_metrics",
    [{}, NEW_METRIC, OTHER_KIND, OTHER_STATUS, OTHER_ERROR, NONE_ERROR, OTHER_COUNT],
)
def test_collect_cache_metrics(old_metrics: EntriesTotalByKindStatusAndErrorCode) -> None:
    for (kind, http_status, error_code), total in old_metrics.items():
        CacheTotalMetricDocument(kind=kind, http_status=http_status, error_code=error_code, total=total).save()

    with patch(
        "cache_maintenance.cache_metrics.get_responses_count_by_kind_status_and_error_code",
        return_value=NEW_METRIC,
    ):
        collect_cache_metrics()

    assert {
        (metric.kind, metric.http_status, metric.error_code): metric.total
        for metric in CacheTotalMetricDocument.objects()
    } == NEW_METRIC
