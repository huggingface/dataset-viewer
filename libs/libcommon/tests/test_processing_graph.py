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


def assert_lists_are_equal(a: List[ProcessingStep], b: List[str]) -> None:
    assert sorted(processing_step.name for processing_step in a) == sorted(b)


def assert_step(
    graph: ProcessingGraph,
    processing_step_name: str,
    children: List[str],
    parents: List[str],
    ancestors: List[str],
) -> None:
    assert_lists_are_equal(graph.get_children(processing_step_name), children)
    assert_lists_are_equal(graph.get_parents(processing_step_name), parents)
    assert_lists_are_equal(graph.get_ancestors(processing_step_name), ancestors)


def test_graph() -> None:
    a = "step_a"
    b = "step_b"
    c = "step_c"
    d = "step_d"
    e = "step_e"
    f = "step_f"
    specification: ProcessingGraphSpecification = {
        a: {"input_type": "dataset", "job_runner_version": 1},
        b: {"input_type": "dataset", "job_runner_version": 1},
        c: {"input_type": "dataset", "triggered_by": a, "job_runner_version": 1},
        d: {"input_type": "dataset", "triggered_by": [a, c], "job_runner_version": 1},
        e: {"input_type": "dataset", "triggered_by": [c], "job_runner_version": 1},
        f: {"input_type": "dataset", "triggered_by": [a, b], "job_runner_version": 1},
    }
    graph = ProcessingGraph(ProcessingGraphConfig(specification).specification)

    assert_step(graph, a, children=[c, d, f], parents=[], ancestors=[])
    assert_step(graph, b, children=[f], parents=[], ancestors=[])
    assert_step(graph, c, children=[d, e], parents=[a], ancestors=[a])
    assert_step(graph, d, children=[], parents=[a, c], ancestors=[a, c])
    assert_step(graph, e, children=[], parents=[c], ancestors=[a, c])
    assert_step(graph, f, children=[], parents=[a, b], ancestors=[a, b])


@pytest.fixture(scope="module")
def graph() -> ProcessingGraph:
    config = ProcessingGraphConfig()
    return ProcessingGraph(config.specification)


@pytest.mark.parametrize(
    "processing_step_name,children,parents,ancestors",
    [
        (
            "dataset-config-names",
            [
                "config-split-names-from-streaming",
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
            ["dataset-config-names"],
            ["dataset-config-names"],
        ),
        (
            "config-split-names-from-info",
            [
                "config-opt-in-out-urls-count",
                "split-first-rows-from-streaming",
                "dataset-split-names",
                "split-duckdb-index",
            ],
            ["config-info"],
            ["dataset-config-names", "config-parquet-and-info", "config-info"],
        ),
        (
            "config-split-names-from-streaming",
            [
                "split-first-rows-from-streaming",
                "dataset-split-names",
                "config-opt-in-out-urls-count",
            ],
            ["dataset-config-names"],
            ["dataset-config-names"],
        ),
        (
            "dataset-split-names",
            ["dataset-is-valid"],
            [
                "dataset-config-names",
                "config-split-names-from-info",
                "config-split-names-from-streaming",
            ],
            [
                "dataset-config-names",
                "config-parquet-and-info",
                "config-info",
                "config-split-names-from-info",
                "config-split-names-from-streaming",
            ],
        ),
        (
            "split-first-rows-from-parquet",
            ["dataset-is-valid", "split-image-url-columns"],
            ["config-parquet"],
            ["config-parquet", "dataset-config-names", "config-parquet-and-info"],
        ),
        (
            "split-first-rows-from-streaming",
            ["dataset-is-valid", "split-image-url-columns"],
            [
                "config-split-names-from-streaming",
                "config-split-names-from-info",
            ],
            [
                "dataset-config-names",
                "config-split-names-from-streaming",
                "config-split-names-from-info",
                "config-parquet-and-info",
                "config-info",
            ],
        ),
        (
            "config-parquet",
            ["config-parquet-metadata", "split-first-rows-from-parquet", "dataset-parquet"],
            ["config-parquet-and-info"],
            ["dataset-config-names", "config-parquet-and-info"],
        ),
        (
            "config-parquet-metadata",
            [],
            ["config-parquet"],
            ["dataset-config-names", "config-parquet-and-info", "config-parquet"],
        ),
        (
            "dataset-parquet",
            [],
            ["dataset-config-names", "config-parquet"],
            ["dataset-config-names", "config-parquet-and-info", "config-parquet"],
        ),
        (
            "config-info",
            ["dataset-info", "config-split-names-from-info"],
            ["config-parquet-and-info"],
            ["dataset-config-names", "config-parquet-and-info"],
        ),
        (
            "dataset-info",
            [],
            ["dataset-config-names", "config-info"],
            ["dataset-config-names", "config-parquet-and-info", "config-info"],
        ),
        (
            "config-size",
            ["dataset-size"],
            ["config-parquet-and-info"],
            ["dataset-config-names", "config-parquet-and-info"],
        ),
        (
            "dataset-size",
            [],
            ["dataset-config-names", "config-size"],
            ["dataset-config-names", "config-parquet-and-info", "config-size"],
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
                "dataset-config-names",
                "config-parquet-and-info",
                "dataset-split-names",
                "config-info",
                "config-parquet",
                "config-split-names-from-info",
                "config-split-names-from-streaming",
                "split-first-rows-from-parquet",
                "split-first-rows-from-streaming",
            ],
        ),
        (
            "split-image-url-columns",
            ["split-opt-in-out-urls-scan"],
            ["split-first-rows-from-streaming", "split-first-rows-from-parquet"],
            [
                "dataset-config-names",
                "config-split-names-from-streaming",
                "config-split-names-from-info",
                "config-info",
                "config-parquet-and-info",
                "split-first-rows-from-streaming",
                "config-parquet",
                "split-first-rows-from-parquet",
            ],
        ),
        (
            "split-opt-in-out-urls-scan",
            ["split-opt-in-out-urls-count"],
            ["split-image-url-columns"],
            [
                "dataset-config-names",
                "config-split-names-from-streaming",
                "config-split-names-from-info",
                "config-info",
                "config-parquet-and-info",
                "split-first-rows-from-streaming",
                "config-parquet",
                "split-first-rows-from-parquet",
                "split-image-url-columns",
            ],
        ),
        (
            "split-opt-in-out-urls-count",
            ["config-opt-in-out-urls-count"],
            ["split-opt-in-out-urls-scan"],
            [
                "dataset-config-names",
                "config-split-names-from-streaming",
                "split-first-rows-from-streaming",
                "config-split-names-from-info",
                "config-info",
                "config-parquet-and-info",
                "split-opt-in-out-urls-scan",
                "config-parquet",
                "split-first-rows-from-parquet",
                "split-image-url-columns",
            ],
        ),
        (
            "config-opt-in-out-urls-count",
            ["dataset-opt-in-out-urls-count"],
            ["split-opt-in-out-urls-count", "config-split-names-from-info", "config-split-names-from-streaming"],
            [
                "dataset-config-names",
                "config-split-names-from-streaming",
                "split-first-rows-from-streaming",
                "config-split-names-from-info",
                "config-info",
                "config-parquet-and-info",
                "split-opt-in-out-urls-count",
                "split-opt-in-out-urls-scan",
                "config-parquet",
                "split-first-rows-from-parquet",
                "split-image-url-columns",
            ],
        ),
        (
            "dataset-opt-in-out-urls-count",
            [],
            ["config-opt-in-out-urls-count", "dataset-config-names"],
            [
                "dataset-config-names",
                "config-split-names-from-streaming",
                "split-first-rows-from-streaming",
                "config-split-names-from-info",
                "config-info",
                "config-parquet-and-info",
                "config-opt-in-out-urls-count",
                "split-opt-in-out-urls-count",
                "split-opt-in-out-urls-scan",
                "config-parquet",
                "split-first-rows-from-parquet",
                "split-image-url-columns",
            ],
        ),
        (
            "split-duckdb-index",
            [],
            ["config-split-names-from-info"],
            [
                "config-split-names-from-info",
                "config-parquet-and-info",
                "config-info",
                "dataset-config-names",
            ],
        ),
    ],
)
def test_default_graph_steps(
    graph: ProcessingGraph, processing_step_name: str, children: List[str], parents: List[str], ancestors: List[str]
) -> None:
    assert_step(graph, processing_step_name, children=children, parents=parents, ancestors=ancestors)


def test_default_graph_first_steps(graph: ProcessingGraph) -> None:
    roots = ["dataset-config-names"]
    assert_lists_are_equal(graph.get_first_processing_steps(), roots)


def test_default_graph_enables_preview(graph: ProcessingGraph) -> None:
    enables_preview = ["split-first-rows-from-streaming", "split-first-rows-from-parquet"]
    assert_lists_are_equal(graph.get_processing_steps_enables_preview(), enables_preview)


def test_default_graph_enables_viewer(graph: ProcessingGraph) -> None:
    enables_viewer = ["config-size"]
    assert_lists_are_equal(graph.get_processing_steps_enables_viewer(), enables_viewer)


def test_default_graph_provide_dataset_config_names(graph: ProcessingGraph) -> None:
    assert_lists_are_equal(graph.get_dataset_config_names_processing_steps(), ["dataset-config-names"])


def test_default_graph_provide_config_split_names(graph: ProcessingGraph) -> None:
    assert_lists_are_equal(
        graph.get_config_split_names_processing_steps(),
        ["config-split-names-from-streaming", "config-split-names-from-info"],
    )
