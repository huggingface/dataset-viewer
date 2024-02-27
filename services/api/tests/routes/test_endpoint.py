# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.


from unittest.mock import patch

import pytest
from libcommon.processing_graph import processing_graph
from pytest import raises

from api.config import EndpointConfig
from api.routes.endpoint import EndpointsDefinition, truncate_features_from_croissant_response


def test_endpoints_definition() -> None:
    endpoint_config = EndpointConfig.from_env()

    endpoints_definition = EndpointsDefinition(processing_graph, endpoint_config)
    assert endpoints_definition

    definition = endpoints_definition.step_by_input_type_and_endpoint
    assert definition

    splits = definition["/splits"]
    assert splits is not None
    assert sorted(list(splits)) == ["config", "dataset"]
    assert splits["dataset"] is not None
    assert splits["config"] is not None

    first_rows = definition["/first-rows"]
    assert first_rows is not None
    assert sorted(list(first_rows)) == ["split"]
    assert first_rows["split"] is not None

    parquet = definition["/parquet"]
    assert parquet is not None
    assert sorted(list(parquet)) == ["config", "dataset"]
    assert parquet["dataset"] is not None
    assert parquet["config"] is not None

    dataset_info = definition["/info"]
    assert dataset_info is not None
    assert sorted(list(dataset_info)) == ["config", "dataset"]
    assert dataset_info["dataset"] is not None
    assert dataset_info["config"] is not None

    size = definition["/size"]
    assert size is not None
    assert sorted(list(size)) == ["config", "dataset"]
    assert size["dataset"] is not None
    assert size["config"] is not None

    opt_in_out_urls = definition["/opt-in-out-urls"]
    assert opt_in_out_urls is not None
    assert sorted(list(opt_in_out_urls)) == ["config", "dataset", "split"]
    assert opt_in_out_urls["split"] is not None
    assert opt_in_out_urls["config"] is not None
    assert opt_in_out_urls["dataset"] is not None

    is_valid = definition["/is-valid"]
    assert is_valid is not None
    assert sorted(list(is_valid)) == ["config", "dataset", "split"]
    assert is_valid["dataset"] is not None
    assert is_valid["config"] is not None
    assert is_valid["split"] is not None

    croissant = definition["/croissant"]
    assert croissant is not None
    assert sorted(list(croissant)) == ["dataset"]
    assert croissant["dataset"] is not None

    # assert old deleted endpoints don't exist
    with raises(KeyError):
        _ = definition["/dataset-info"]
    with raises(KeyError):
        _ = definition["/parquet-and-dataset-info"]
    with raises(KeyError):
        _ = definition["/config-names"]


MAX_COLUMNS = 3


@pytest.mark.parametrize("num_columns", [1, 3])
def test_truncate_features_from_croissant_response(num_columns: int) -> None:
    content = {
        "croissant": {
            "recordSet": [
                {
                    "field": [{"name": f"col_{i}", "type": "string"} for i in range(num_columns)],
                    "description": "description",
                }
            ]
        }
    }
    with patch("api.routes.endpoint.MAX_COLUMNS", 2):
        truncate_features_from_croissant_response(content)
    if num_columns <= 2:
        assert len(content["croissant"]["recordSet"][0]["field"]) == num_columns
    else:
        assert len(content["croissant"]["recordSet"][0]["field"]) == 2
        assert "max number of columns reached" in content["croissant"]["recordSet"][0]["description"]
