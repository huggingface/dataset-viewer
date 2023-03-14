# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingGraph


def test_default_graph() -> None:
    config = ProcessingGraphConfig()
    graph = ProcessingGraph(config.specification)

    config_names = graph.get_step("/config-names")
    split_names_from_streaming = graph.get_step("/split-names-from-streaming")
    splits = graph.get_step("/splits")
    first_rows = graph.get_step("/first-rows")
    parquet_and_dataset_info = graph.get_step("/parquet-and-dataset-info")
    parquet = graph.get_step("/parquet")
    dataset_info = graph.get_step("/dataset-info")
    config_size = graph.get_step("config-size")
    dataset_size = graph.get_step("dataset-size")
    split_names_from_dataset_info = graph.get_step("/split-names-from-dataset-info")
    dataset_split_names = graph.get_step("dataset-split-names")

    assert config_names is not None
    assert config_names.parent is None
    assert config_names.children == [split_names_from_streaming]
    assert config_names.get_ancestors() == []

    assert split_names_from_streaming is not None
    assert split_names_from_streaming.parent is config_names
    assert split_names_from_streaming.children == [first_rows, dataset_split_names]
    assert split_names_from_streaming.get_ancestors() == [config_names]

    assert splits is not None
    assert splits.parent is None
    assert splits.children == []
    assert splits.get_ancestors() == []

    assert first_rows is not None
    assert first_rows.parent is split_names_from_streaming
    assert first_rows.children == []
    assert first_rows.get_ancestors() == [config_names, split_names_from_streaming]

    assert parquet_and_dataset_info is not None
    assert parquet_and_dataset_info.parent is None
    assert parquet_and_dataset_info.children == [parquet, dataset_info, config_size]
    assert parquet_and_dataset_info.get_ancestors() == []

    assert parquet is not None
    assert parquet.parent is parquet_and_dataset_info
    assert parquet.children == []
    assert parquet.get_ancestors() == [parquet_and_dataset_info]

    assert dataset_info is not None
    assert dataset_info.parent is parquet_and_dataset_info
    assert dataset_info.children == [split_names_from_dataset_info]
    assert dataset_info.get_ancestors() == [parquet_and_dataset_info]

    assert split_names_from_dataset_info is not None
    assert split_names_from_dataset_info.parent is dataset_info
    assert split_names_from_dataset_info.children == []
    assert split_names_from_dataset_info.get_ancestors() == [parquet_and_dataset_info, dataset_info]

    assert config_size is not None
    assert config_size.parent is parquet_and_dataset_info
    assert config_size.children == [dataset_size]
    assert config_size.get_ancestors() == [parquet_and_dataset_info]

    assert dataset_size is not None
    assert dataset_size.parent is config_size
    assert dataset_size.children == []
    assert dataset_size.get_ancestors() == [parquet_and_dataset_info, config_size]

    assert dataset_split_names is not None
    assert dataset_split_names.parent is split_names_from_streaming
    assert dataset_split_names.children == []
    assert dataset_split_names.get_ancestors() == [config_names, split_names_from_streaming]

    assert graph.get_first_steps() == [config_names, splits, parquet_and_dataset_info]
    assert graph.get_steps_required_by_dataset_viewer() == [splits, first_rows]
