# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.config import ProcessingGraphConfig


def test_default_graph():
    config = ProcessingGraphConfig()
    graph = config.graph

    splits = graph.get_step("/splits")
    first_rows = graph.get_step("/first-rows")

    assert splits is not None
    assert first_rows is not None

    assert splits.parent is None
    assert first_rows.parent is splits

    assert splits.children == [first_rows]
    assert first_rows.children == []

    assert splits.get_ancestors() == []
    assert first_rows.get_ancestors() == [splits]

    assert graph.get_first_steps() == [splits]
    assert graph.get_steps_required_by_dataset_viewer() == [splits, first_rows]
