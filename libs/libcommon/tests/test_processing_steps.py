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
    split_first_rows_from_streaming = graph.get_step("split-first-rows-from-streaming")
    parquet_and_dataset_info = graph.get_step("/parquet-and-dataset-info")
    config_parquet = graph.get_step("config-parquet")
    dataset_parquet = graph.get_step("dataset-parquet")
    dataset_info = graph.get_step("/dataset-info")
    config_size = graph.get_step("config-size")
    dataset_size = graph.get_step("dataset-size")
    split_names_from_dataset_info = graph.get_step("/split-names-from-dataset-info")
    dataset_split_names_from_streaming = graph.get_step("dataset-split-names-from-streaming")
    dataset_split_names_from_dataset_info = graph.get_step("dataset-split-names-from-dataset-info")

    assert config_names is not None
    assert config_names.parent is None
    assert config_names.children == [split_names_from_streaming]
    assert config_names.get_ancestors() == []

    assert split_names_from_streaming is not None
    assert split_names_from_streaming.parent is config_names
    assert split_names_from_streaming.children == [split_first_rows_from_streaming, dataset_split_names_from_streaming]
    assert split_names_from_streaming.get_ancestors() == [config_names]

    assert splits is not None
    assert splits.parent is None
    assert splits.children == []
    assert splits.get_ancestors() == []

    assert split_first_rows_from_streaming is not None
    assert split_first_rows_from_streaming.parent is split_names_from_streaming
    assert split_first_rows_from_streaming.children == []
    assert split_first_rows_from_streaming.get_ancestors() == [config_names, split_names_from_streaming]

    assert parquet_and_dataset_info is not None
    assert parquet_and_dataset_info.parent is None
    assert parquet_and_dataset_info.children == [config_parquet, dataset_info, config_size]
    assert parquet_and_dataset_info.get_ancestors() == []

    assert config_parquet is not None
    assert config_parquet.parent is parquet_and_dataset_info
    assert config_parquet.children == [dataset_parquet]
    assert config_parquet.get_ancestors() == [parquet_and_dataset_info]

    assert dataset_parquet is not None
    assert dataset_parquet.parent is config_parquet
    assert dataset_parquet.children == []
    assert dataset_parquet.get_ancestors() == [parquet_and_dataset_info, config_parquet]

    assert dataset_info is not None
    assert dataset_info.parent is parquet_and_dataset_info
    assert dataset_info.children == [split_names_from_dataset_info]
    assert dataset_info.get_ancestors() == [parquet_and_dataset_info]

    assert split_names_from_dataset_info is not None
    assert split_names_from_dataset_info.parent is dataset_info
    assert split_names_from_dataset_info.children == [dataset_split_names_from_dataset_info]
    assert split_names_from_dataset_info.get_ancestors() == [parquet_and_dataset_info, dataset_info]

    assert config_size is not None
    assert config_size.parent is parquet_and_dataset_info
    assert config_size.children == [dataset_size]
    assert config_size.get_ancestors() == [parquet_and_dataset_info]

    assert dataset_size is not None
    assert dataset_size.parent is config_size
    assert dataset_size.children == []
    assert dataset_size.get_ancestors() == [parquet_and_dataset_info, config_size]

    assert dataset_split_names_from_streaming is not None
    assert dataset_split_names_from_streaming.parent is split_names_from_streaming
    assert dataset_split_names_from_streaming.children == []
    assert dataset_split_names_from_streaming.get_ancestors() == [config_names, split_names_from_streaming]

    assert dataset_split_names_from_dataset_info is not None
    assert dataset_split_names_from_dataset_info.parent is split_names_from_dataset_info
    assert dataset_split_names_from_dataset_info.children == []
    assert dataset_split_names_from_dataset_info.get_ancestors() == [
        parquet_and_dataset_info,
        dataset_info,
        split_names_from_dataset_info,
    ]

    assert graph.get_first_steps() == [config_names, splits, parquet_and_dataset_info]
    assert graph.get_steps_required_by_dataset_viewer() == [splits, split_first_rows_from_streaming]
