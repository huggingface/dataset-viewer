# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.config import ProcessingGraphConfig


def test_default_graph():
    config = ProcessingGraphConfig()
    graph = config.graph

    splits = graph.get_step("/splits")
    first_rows = graph.get_step("/first-rows")
    parquet = graph.get_step("/parquet")
    size = graph.get_step("/size")

    assert splits is not None
    assert first_rows is not None
    assert parquet is not None
    assert size is not None

    assert splits.parent is None
    assert first_rows.parent is splits
    assert parquet.parent is None
    assert size.parent is parquet

    assert splits.children == [first_rows]
    assert first_rows.children == []
    assert parquet.children == [size]
    assert size.children == []

    assert splits.get_ancestors() == []
    assert first_rows.get_ancestors() == [splits]
    assert parquet.get_ancestors() == []
    assert size.get_ancestors() == [parquet]

    assert graph.get_first_steps() == [splits, parquet]
    assert graph.get_steps_required_by_dataset_viewer() == [splits, first_rows]
