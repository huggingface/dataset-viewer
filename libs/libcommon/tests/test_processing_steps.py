# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List, Optional

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingGraph, ProcessingStep


def get_step_name(step: ProcessingStep) -> str:
    return step.name


def assert_lists_are_equal(a: List[ProcessingStep], b: List[ProcessingStep]) -> None:
    assert sorted(a, key=get_step_name) == sorted(b, key=get_step_name)


def assert_step(
    step: ProcessingStep,
    parent: Optional[ProcessingStep],
    children: List[ProcessingStep],
    ancestors: List[ProcessingStep],
) -> None:
    assert step is not None
    assert step.parent is parent
    assert_lists_are_equal(step.children, children)
    assert_lists_are_equal(step.get_ancestors(), ancestors)


def test_default_graph() -> None:
    config = ProcessingGraphConfig()
    graph = ProcessingGraph(config.specification)

    config_names = graph.get_step("/config-names")
    split_names_from_streaming = graph.get_step("/split-names-from-streaming")
    splits = graph.get_step("/splits")
    split_first_rows_from_streaming = graph.get_step("split-first-rows-from-streaming")
    parquet_and_dataset_info = graph.get_step("/parquet-and-dataset-info")
    config_parquet_and_info = graph.get_step("config-parquet-and-info")
    config_parquet = graph.get_step("config-parquet")
    dataset_parquet = graph.get_step("dataset-parquet")
    config_info = graph.get_step("config-info")
    dataset_info = graph.get_step("dataset-info")
    config_size = graph.get_step("config-size")
    dataset_size = graph.get_step("dataset-size")
    split_names_from_dataset_info = graph.get_step("/split-names-from-dataset-info")
    dataset_split_names_from_streaming = graph.get_step("dataset-split-names-from-streaming")
    dataset_split_names_from_dataset_info = graph.get_step("dataset-split-names-from-dataset-info")
    split_first_rows_from_parquet = graph.get_step("split-first-rows-from-parquet")

    assert_step(
        config_names, parent=None, children=[split_names_from_streaming, config_parquet_and_info], ancestors=[]
    )
    assert_step(
        split_names_from_streaming,
        parent=config_names,
        children=[split_first_rows_from_streaming, dataset_split_names_from_streaming],
        ancestors=[config_names],
    )
    assert_step(splits, parent=None, children=[], ancestors=[])
    assert_step(
        split_first_rows_from_streaming,
        parent=split_names_from_streaming,
        children=[],
        ancestors=[config_names, split_names_from_streaming],
    )
    assert_step(parquet_and_dataset_info, parent=None, children=[], ancestors=[])
    assert_step(
        config_parquet_and_info,
        parent=config_names,
        children=[config_parquet, config_info, config_size],
        ancestors=[config_names],
    )
    assert_step(
        config_parquet,
        parent=config_parquet_and_info,
        children=[split_first_rows_from_parquet, dataset_parquet],
        ancestors=[config_names, config_parquet_and_info],
    )
    assert_step(
        dataset_parquet,
        parent=config_parquet,
        children=[],
        ancestors=[config_names, config_parquet_and_info, config_parquet],
    )
    assert_step(
        config_info,
        parent=config_parquet_and_info,
        children=[dataset_info, split_names_from_dataset_info],
        ancestors=[config_names, config_parquet_and_info],
    )
    assert_step(
        dataset_info, parent=config_info, children=[], ancestors=[config_names, config_parquet_and_info, config_info]
    )
    assert_step(
        config_size,
        parent=config_parquet_and_info,
        children=[dataset_size],
        ancestors=[config_names, config_parquet_and_info],
    )
    assert_step(
        dataset_size, parent=config_size, children=[], ancestors=[config_names, config_parquet_and_info, config_size]
    )
    assert_step(
        split_names_from_dataset_info,
        parent=config_info,
        children=[dataset_split_names_from_dataset_info],
        ancestors=[config_names, config_parquet_and_info, config_info],
    )
    assert_step(
        dataset_split_names_from_streaming,
        parent=split_names_from_streaming,
        children=[],
        ancestors=[config_names, split_names_from_streaming],
    )
    assert_step(
        dataset_split_names_from_dataset_info,
        parent=split_names_from_dataset_info,
        children=[],
        ancestors=[
            config_names,
            config_parquet_and_info,
            config_info,
            split_names_from_dataset_info,
        ],
    )
    assert_step(
        split_first_rows_from_parquet,
        parent=config_parquet,
        children=[],
        ancestors=[config_names, config_parquet_and_info, config_parquet],
    )

    assert graph.get_first_steps() == [config_names, splits, parquet_and_dataset_info]
    assert graph.get_steps_required_by_dataset_viewer() == [splits, split_first_rows_from_streaming]
