# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 The HuggingFace Authors.


from typing import Any
from unittest.mock import patch

import pytest
from datasets import List, Value

from libcommon.croissant_utils import (
    escape_ids,
    escape_jsonpath_key,
    feature_to_croissant_field,
    truncate_features_from_croissant_crumbs_response,
)


@pytest.mark.parametrize("num_columns", [1, 3])
def test_truncate_features_from_croissant_crumbs_response(num_columns: int) -> None:
    content = {
        "recordSet": [
            {
                "field": [{"name": f"col_{i}", "type": "string"} for i in range(num_columns)],
                "description": "description",
            }
        ]
    }
    with patch("libcommon.croissant_utils.MAX_COLUMNS", 2):
        truncate_features_from_croissant_crumbs_response(content)
    if num_columns <= 2:
        assert len(content["recordSet"][0]["field"]) == num_columns
    else:
        assert len(content["recordSet"][0]["field"]) == 2
        assert "max number of columns reached" in content["recordSet"][0]["description"]


@pytest.mark.parametrize(
    "id_to_escape, ids, expected_id",
    [
        ("valid_id", {"other", "other2"}, "valid_id"),
        ("id with spaces", set(), "id_with_spaces"),
        ("a/b/c", set(), "a_b_c"),
        ("a/b/c", {"a_b_c"}, "a_b_c_0"),
        ("a/b/c", {"a_b_c", "a_b_c_0"}, "a_b_c_0_0"),
        ("a@#$b", set(), "a___b"),
        ("", set(), ""),
        ("", {""}, "_0"),
    ],
)
def test_escape_ids(id_to_escape: str, ids: set[str], expected_id: str) -> None:
    """Tests the expected_id function with various inputs."""
    assert escape_ids(id_to_escape, ids=ids.copy()) == expected_id


@pytest.mark.parametrize(
    "feature_name, expected_output",
    [
        ("simple_feature", "simple_feature"),
        ("feature/with/slash", "['feature/with/slash']"),
        ("feature'with'quote", r"['feature\'with\'quote']"),
        ("feature[with]brackets", r"['feature\[with\]brackets']"),
        ("feature[with/slash]'and'quote", r"['feature\[with/slash\]\'and\'quote']"),
        (r"feature\'already\'escaped", r"['feature\'already\'escaped']"),
        ("feature with spaces", "['feature with spaces']"),
    ],
)
def test_escape_jsonpath_key(feature_name: str, expected_output: str) -> None:
    """Tests the escape_jsonpath_key function with various inputs."""
    assert escape_jsonpath_key(feature_name) == expected_output


@pytest.mark.parametrize(
    ("hf_datasets_feature", "croissant_field"),
    [
        (
            Value(dtype="int32"),
            {
                "@type": "cr:Field",
                "@id": "field_name",
                "dataType": "cr:Int32",
                "source": {"fileSet": {"@id": "distribution_name"}, "extract": {"column": "column_name"}},
            },
        ),
        (
            List(Value(dtype="int32")),
            {
                "@type": "cr:Field",
                "@id": "field_name",
                "dataType": "cr:Int32",
                "source": {"fileSet": {"@id": "distribution_name"}, "extract": {"column": "column_name"}},
                "isArray": True,
                "arrayShape": "-1",
            },
        ),
        (
            List(List(Value(dtype="int32"), length=3)),
            {
                "@type": "cr:Field",
                "@id": "field_name",
                "dataType": "cr:Int32",
                "source": {"fileSet": {"@id": "distribution_name"}, "extract": {"column": "column_name"}},
                "isArray": True,
                "arrayShape": "-1,3",
            },
        ),
        (
            List({"sub-field": {"sub-sub-field": Value(dtype="int32")}}),
            {
                "@type": "cr:Field",
                "@id": "field_name",
                "subField": [
                    {
                        "@type": "cr:Field",
                        "@id": "field_name/sub-field",
                        "subField": [
                            {
                                "@type": "cr:Field",
                                "@id": "field_name/sub-field/sub-sub-field",
                                "dataType": "cr:Int32",
                                "source": {
                                    "fileSet": {"@id": "distribution_name"},
                                    "extract": {"column": "column_name"},
                                    "transform": [{"jsonPath": "sub-field"}, {"jsonPath": "sub-sub-field"}],
                                },
                            }
                        ],
                    }
                ],
                "isArray": True,
                "arrayShape": "-1",
            },
        ),
    ],
)
def test_feature_to_croissant_field(hf_datasets_feature: Any, croissant_field: Any) -> None:
    assert (
        feature_to_croissant_field("distribution_name", "field_name", "column_name", hf_datasets_feature, existing_ids=[])
        == croissant_field
    )
