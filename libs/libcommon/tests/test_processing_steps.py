# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from typing import List

import pytest

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


@pytest.fixture(scope="module")
def graph() -> ProcessingGraph:
    config = ProcessingGraphConfig()
    return ProcessingGraph(config.specification)


@pytest.mark.parametrize(
    "step_name,children,ancestors",
    [
        ("/config-names", ["/split-names-from-streaming"], []),
        (
            "/split-names-from-dataset-info",
            ["dataset-split-names-from-dataset-info", "split-first-rows-from-streaming", "dataset-split-names"],
            ["/parquet-and-dataset-info", "config-info"],
        ),
        (
            "/split-names-from-streaming",
            ["split-first-rows-from-streaming", "dataset-split-names-from-streaming", "dataset-split-names"],
            ["/config-names"],
        ),
        (
            "dataset-split-names-from-dataset-info",
            [],
            ["/parquet-and-dataset-info", "config-info", "/split-names-from-dataset-info"],
        ),
        ("dataset-split-names-from-streaming", [], ["/config-names", "/split-names-from-streaming"]),
        (
            "dataset-split-names",
            [],
            [
                "/parquet-and-dataset-info",
                "config-info",
                "/split-names-from-dataset-info",
                "/config-names",
                "/split-names-from-streaming",
            ],
        ),
        ("split-first-rows-from-parquet", [], ["config-parquet", "/parquet-and-dataset-info"]),
        (
            "split-first-rows-from-streaming",
            [],
            [
                "/config-names",
                "/split-names-from-streaming",
                "/split-names-from-dataset-info",
                "/parquet-and-dataset-info",
                "config-info",
            ],
        ),
        ("/parquet-and-dataset-info", ["config-parquet", "config-info", "config-size"], []),
        ("config-parquet", ["split-first-rows-from-parquet", "dataset-parquet"], ["/parquet-and-dataset-info"]),
        ("dataset-parquet", [], ["/parquet-and-dataset-info", "config-parquet"]),
        ("config-info", ["dataset-info", "/split-names-from-dataset-info"], ["/parquet-and-dataset-info"]),
        ("dataset-info", [], ["/parquet-and-dataset-info", "config-info"]),
        ("config-size", ["dataset-size"], ["/parquet-and-dataset-info"]),
        ("dataset-size", [], ["/parquet-and-dataset-info", "config-size"]),
    ],
)
def test_default_graph_steps(
    graph: ProcessingGraph, step_name: str, children: List[str], ancestors: List[str]
) -> None:
    assert_step(
        graph.get_step(step_name),
        children=[graph.get_step(child) for child in children],
        ancestors=[graph.get_step(ancestor) for ancestor in ancestors],
    )


def test_default_graph_first_steps(graph: ProcessingGraph) -> None:
    assert_lists_are_equal(
        graph.get_first_steps(),
        [graph.get_step(step_name) for step_name in {"/config-names", "/parquet-and-dataset-info"}],
    )


def test_default_graph_required_by_dataset_viewer(graph: ProcessingGraph) -> None:
    assert_lists_are_equal(
        graph.get_steps_required_by_dataset_viewer(),
        [graph.get_step(step_name) for step_name in {"split-first-rows-from-streaming"}],
    )
