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
    parents: List[ProcessingStep],
    ancestors: List[ProcessingStep],
) -> None:
    assert step is not None
    assert_lists_are_equal(step.children, children)
    assert_lists_are_equal(step.parents, parents)
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

    assert_step(a, children=[c, d, f], parents=[], ancestors=[])
    assert_step(b, children=[f], parents=[], ancestors=[])
    assert_step(c, children=[d, e], parents=[a], ancestors=[a])
    assert_step(d, children=[], parents=[a, c], ancestors=[a, c])
    assert_step(e, children=[], parents=[c], ancestors=[a, c])
    assert_step(f, children=[], parents=[a, b], ancestors=[a, b])


@pytest.fixture(scope="module")
def graph() -> ProcessingGraph:
    config = ProcessingGraphConfig()
    return ProcessingGraph(config.specification)


@pytest.mark.parametrize(
    "step_name,children,parents,ancestors",
    [
        (
            "/config-names",
            [
                "/split-names-from-streaming",
                "config-parquet-and-info",
                "dataset-opt-in-out-urls-count",
                "dataset-split-names",
                "dataset-parquet",
                "dataset-info",
                "dataset-size",
            ],
            [],
            [],
        ),
        (
            "config-parquet-and-info",
            [
                "config-parquet",
                "config-info",
                "config-size",
            ],
            ["/config-names"],
            ["/config-names"],
        ),
        (
            "/split-names-from-dataset-info",
            [
                "split-first-rows-from-streaming",
                "dataset-split-names",
            ],
            ["config-info"],
            ["/config-names", "config-parquet-and-info", "config-info"],
        ),
        (
            "/split-names-from-streaming",
            ["split-first-rows-from-streaming", "dataset-split-names", "config-opt-in-out-urls-count"],
            ["/config-names"],
            ["/config-names"],
        ),
        (
            "dataset-split-names",
            ["dataset-is-valid"],
            [
                "/config-names",
                "/split-names-from-dataset-info",
                "/split-names-from-streaming",
            ],
            [
                "/config-names",
                "config-parquet-and-info",
                "config-info",
                "/split-names-from-dataset-info",
                "/split-names-from-streaming",
            ],
        ),
        (
            "split-first-rows-from-parquet",
            ["dataset-is-valid"],
            ["config-parquet"],
            ["config-parquet", "/config-names", "config-parquet-and-info"],
        ),
        (
            "split-first-rows-from-streaming",
            ["dataset-is-valid", "split-opt-in-out-urls-scan"],
            [
                "/split-names-from-streaming",
                "/split-names-from-dataset-info",
            ],
            [
                "/config-names",
                "/split-names-from-streaming",
                "/split-names-from-dataset-info",
                "config-parquet-and-info",
                "config-info",
            ],
        ),
        (
            "config-parquet",
            ["split-first-rows-from-parquet", "dataset-parquet"],
            ["config-parquet-and-info"],
            ["/config-names", "config-parquet-and-info"],
        ),
        (
            "dataset-parquet",
            [],
            ["/config-names", "config-parquet"],
            ["/config-names", "config-parquet-and-info", "config-parquet"],
        ),
        (
            "config-info",
            ["dataset-info", "/split-names-from-dataset-info"],
            ["config-parquet-and-info"],
            ["/config-names", "config-parquet-and-info"],
        ),
        (
            "dataset-info",
            [],
            ["/config-names", "config-info"],
            ["/config-names", "config-parquet-and-info", "config-info"],
        ),
        ("config-size", ["dataset-size"], ["config-parquet-and-info"], ["/config-names", "config-parquet-and-info"]),
        (
            "dataset-size",
            [],
            ["/config-names", "config-size"],
            ["/config-names", "config-parquet-and-info", "config-size"],
        ),
        (
            "dataset-is-valid",
            [],
            [
                "dataset-split-names",
                "split-first-rows-from-parquet",
                "split-first-rows-from-streaming",
            ],
            [
                "/config-names",
                "config-parquet-and-info",
                "dataset-split-names",
                "config-info",
                "config-parquet",
                "/split-names-from-dataset-info",
                "/split-names-from-streaming",
                "split-first-rows-from-parquet",
                "split-first-rows-from-streaming",
            ],
        ),
        (
            "split-opt-in-out-urls-scan",
            ["split-opt-in-out-urls-count"],
            ["split-first-rows-from-streaming"],
            [
                "/config-names",
                "/split-names-from-streaming",
                "split-first-rows-from-streaming",
                "/split-names-from-dataset-info",
                "config-info",
                "config-parquet-and-info",
            ],
        ),
        (
            "split-opt-in-out-urls-count",
            ["config-opt-in-out-urls-count"],
            ["split-opt-in-out-urls-scan"],
            [
                "/config-names",
                "/split-names-from-streaming",
                "split-first-rows-from-streaming",
                "/split-names-from-dataset-info",
                "config-info",
                "config-parquet-and-info",
                "split-opt-in-out-urls-scan",
            ],
        ),
        (
            "config-opt-in-out-urls-count",
            ["dataset-opt-in-out-urls-count"],
            ["split-opt-in-out-urls-count", "/split-names-from-streaming"],
            [
                "/config-names",
                "/split-names-from-streaming",
                "split-first-rows-from-streaming",
                "/split-names-from-dataset-info",
                "config-info",
                "config-parquet-and-info",
                "split-opt-in-out-urls-count",
                "split-opt-in-out-urls-scan",
            ],
        ),
        (
            "dataset-opt-in-out-urls-count",
            [],
            ["config-opt-in-out-urls-count", "/config-names"],
            [
                "/config-names",
                "/split-names-from-streaming",
                "split-first-rows-from-streaming",
                "/split-names-from-dataset-info",
                "config-info",
                "config-parquet-and-info",
                "config-opt-in-out-urls-count",
                "split-opt-in-out-urls-count",
                "split-opt-in-out-urls-scan",
            ],
        ),
    ],
)
def test_default_graph_steps(
    graph: ProcessingGraph, step_name: str, children: List[str], parents: List[str], ancestors: List[str]
) -> None:
    assert_step(
        graph.get_step(step_name),
        children=[graph.get_step(child) for child in children],
        parents=[graph.get_step(parent) for parent in parents],
        ancestors=[graph.get_step(ancestor) for ancestor in ancestors],
    )


def test_default_graph_first_steps(graph: ProcessingGraph) -> None:
    assert_lists_are_equal(
        graph.get_first_steps(),
        [graph.get_step(step_name) for step_name in {"/config-names"}],
    )


def test_default_graph_required_by_dataset_viewer(graph: ProcessingGraph) -> None:
    assert_lists_are_equal(
        graph.get_steps_required_by_dataset_viewer(),
        [graph.get_step(step_name) for step_name in {"split-first-rows-from-streaming"}],
    )


def test_default_graph_provide_dataset_config_names(graph: ProcessingGraph) -> None:
    assert_lists_are_equal(
        graph.provide_dataset_config_names,
        [graph.get_step(step_name) for step_name in {"/config-names"}],
    )


def test_default_graph_provide_config_split_names(graph: ProcessingGraph) -> None:
    assert_lists_are_equal(
        graph.provide_config_split_names,
        [graph.get_step(step_name) for step_name in {"/split-names-from-streaming", "/split-names-from-dataset-info"}],
    )
