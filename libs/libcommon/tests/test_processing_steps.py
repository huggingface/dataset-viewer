# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import (
    ProcessingGraph,
    ProcessingGraphSpecification,
    ProcessingStep,
)


def get_step_name(step: ProcessingStep) -> str:
    return step.name


def assert_lists_are_equal(a: List[ProcessingStep], b: List[ProcessingStep]) -> None:
    assert sorted(a, key=get_step_name) == sorted(b, key=get_step_name)


def assert_step(
    step: ProcessingStep,
    children: List[ProcessingStep],
    ancestors: List[ProcessingStep],
) -> None:
    assert step is not None
    assert_lists_are_equal(step.children, children)
    assert_lists_are_equal(step.get_ancestors(), ancestors)


def test_graph() -> None:
    specification: ProcessingGraphSpecification = {
        "a": {"input_type": "dataset", "job_runner_version": 1},
        "b": {"input_type": "dataset", "job_runner_version": 1},
        "c": {"input_type": "dataset", "requires": "a", "job_runner_version": 1},
        "d": {"input_type": "dataset", "requires": ["a", "c"], "job_runner_version": 1},
        "e": {"input_type": "dataset", "requires": ["c"], "job_runner_version": 1},
        "f": {"input_type": "dataset", "requires": ["a", "b"], "job_runner_version": 1},
    }
    graph = ProcessingGraph(ProcessingGraphConfig(specification).specification)
    a = graph.get_step("a")
    b = graph.get_step("b")
    c = graph.get_step("c")
    d = graph.get_step("d")
    e = graph.get_step("e")
    f = graph.get_step("f")

    assert_step(a, children=[c, f], ancestors=[])
    assert_step(b, children=[f], ancestors=[])
    assert_step(c, children=[d, e], ancestors=[a])
    assert_step(d, children=[], ancestors=[a, c])
    assert_step(e, children=[], ancestors=[a, c])
    assert_step(f, children=[], ancestors=[a, b])


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
    config_info = graph.get_step("config-info")
    dataset_info = graph.get_step("dataset-info")
    config_size = graph.get_step("config-size")
    dataset_size = graph.get_step("dataset-size")
    split_names_from_dataset_info = graph.get_step("/split-names-from-dataset-info")
    dataset_split_names_from_streaming = graph.get_step("dataset-split-names-from-streaming")
    dataset_split_names_from_dataset_info = graph.get_step("dataset-split-names-from-dataset-info")
    split_first_rows_from_parquet = graph.get_step("split-first-rows-from-parquet")

    assert_step(config_names, children=[split_names_from_streaming], ancestors=[])
    assert_step(
        split_names_from_streaming,
        children=[split_first_rows_from_streaming, dataset_split_names_from_streaming],
        ancestors=[config_names],
    )
    assert_step(splits, children=[], ancestors=[])
    assert_step(
        split_first_rows_from_streaming,
        children=[],
        ancestors=[
            config_names,
            split_names_from_streaming,
            split_names_from_dataset_info,
            parquet_and_dataset_info,
            config_info,
        ],
    )
    assert_step(parquet_and_dataset_info, children=[config_parquet, config_info, config_size], ancestors=[])
    assert_step(
        config_parquet,
        children=[split_first_rows_from_parquet, dataset_parquet],
        ancestors=[parquet_and_dataset_info],
    )
    assert_step(dataset_parquet, children=[], ancestors=[parquet_and_dataset_info, config_parquet])
    assert_step(
        config_info,
        children=[dataset_info, split_names_from_dataset_info],
        ancestors=[parquet_and_dataset_info],
    )
    assert_step(dataset_info, children=[], ancestors=[parquet_and_dataset_info, config_info])
    assert_step(config_size, children=[dataset_size], ancestors=[parquet_and_dataset_info])
    assert_step(dataset_size, children=[], ancestors=[parquet_and_dataset_info, config_size])
    assert_step(
        split_names_from_dataset_info,
        children=[split_first_rows_from_streaming, dataset_split_names_from_dataset_info],
        ancestors=[parquet_and_dataset_info, config_info],
    )
    assert_step(
        dataset_split_names_from_streaming,
        children=[],
        ancestors=[config_names, split_names_from_streaming],
    )
    assert_step(
        dataset_split_names_from_dataset_info,
        children=[],
        ancestors=[parquet_and_dataset_info, config_info, split_names_from_dataset_info],
    )
    assert_step(
        split_first_rows_from_parquet,
        children=[],
        ancestors=[parquet_and_dataset_info, config_parquet],
    )

    assert graph.get_first_steps() == [config_names, splits, parquet_and_dataset_info]
    assert graph.get_steps_required_by_dataset_viewer() == [splits, split_first_rows_from_streaming]
