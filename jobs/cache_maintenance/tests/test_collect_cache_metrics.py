# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from http import HTTPStatus

from libcommon.simple_cache import CacheTotalMetricDocument, upsert_response

from cache_maintenance.cache_metrics import collect_cache_metrics

from .utils import REVISION_NAME


def test_collect_cache_metrics() -> None:
    dataset = "test_dataset"
    config = None
    split = None
    content = {"some": "content"}
    kind = "kind"
    upsert_response(
        kind=kind,
        dataset=dataset,
        dataset_git_revision=REVISION_NAME,
        config=config,
        split=split,
        content=content,
        http_status=HTTPStatus.OK,
    )

    collect_cache_metrics()

    cache_metrics = CacheTotalMetricDocument.objects()
    assert cache_metrics
    assert len(cache_metrics) == 1

    metric = cache_metrics.first()
    assert metric is not None
    assert metric.kind == kind
    assert metric.error_code is None
    assert metric.http_status == HTTPStatus.OK
    assert metric.total == 1
