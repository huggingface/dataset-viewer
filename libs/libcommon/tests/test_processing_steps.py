# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.config import ProcessingGraphConfig


def test_default_graph():
    config = ProcessingGraphConfig()
    graph = config.graph

    splits = graph.get_step("/splits")
    first_rows = graph.get_step("/first-rows")
    parquet_and_dataset_info = graph.get_step("/parquet-and-dataset-info")
    parquet = graph.get_step("/parquet")
    dataset_info = graph.get_step("/dataset-info")
    sizes = graph.get_step("/sizes")

    assert splits is not None
    assert first_rows is not None
    assert parquet_and_dataset_info is not None
    assert parquet is not None
    assert dataset_info is not None
    assert sizes is not None

    assert splits.parent is None
    assert first_rows.parent is splits
    assert parquet_and_dataset_info.parent is None
    assert parquet.parent is parquet_and_dataset_info
    assert dataset_info.parent is parquet_and_dataset_info
    assert sizes.parent is dataset_info

    assert splits.children == [first_rows]
    assert first_rows.children == []
    assert parquet_and_dataset_info.children == [parquet, dataset_info]
    assert parquet.children == []
    assert dataset_info.children == [sizes]
    assert sizes.children == []

    assert splits.get_ancestors() == []
    assert first_rows.get_ancestors() == [splits]
    assert parquet_and_dataset_info.get_ancestors() == []
    assert parquet.get_ancestors() == [parquet_and_dataset_info]
    assert dataset_info.get_ancestors() == [parquet_and_dataset_info]
    assert sizes.get_ancestors() == [parquet_and_dataset_info, dataset_info]

    assert graph.get_first_steps() == [splits, parquet_and_dataset_info]
    assert graph.get_steps_required_by_dataset_viewer() == [splits, first_rows]
