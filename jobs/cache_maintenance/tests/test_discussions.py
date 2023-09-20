# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

import pytest
from cache_maintenance.discussions import limit_to_one_dataset_per_namespace, create_link


@pytest.mark.parametrize(
    "datasets, valid_expected_datasets",
    [
        (set(), [set()]),
        ({"a/b"}, [{"a/b"}]),
        ({"a"}, [set()]),
        ({"a/b/c"}, [set()]),
        ({"a/b", "a/b"}, [{"a/b"}]),
        ({"a/b", "a/c"}, [{"a/b"}, {"a/c"}]),
        ({"a/b", "b/b"}, [{"a/b", "b/b"}]),
        ({"a/b", "b"}, [{"a/b"}]),
    ],
)
def test_limit_to_one_dataset_per_namespace(datasets: set[str], valid_expected_datasets: list[set[str]]) -> None:
    assert any(
        limit_to_one_dataset_per_namespace(datasets=datasets) == expected_datasets
        for expected_datasets in valid_expected_datasets
    )


def test_create_link() -> None:
    assert (
        create_link(dataset="a/b", hf_endpoint="https://huggingface.co", parquet_revision="c/d")
        == "[`c/d`](https://huggingface.co/datasets/a/b/tree/c%2Fd)"
    )
