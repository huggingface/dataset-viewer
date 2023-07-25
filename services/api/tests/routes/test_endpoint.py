# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from libcommon.config import ProcessingGraphConfig
from libcommon.processing_graph import ProcessingGraph
from pytest import raises

from api.config import EndpointConfig
from api.routes.endpoint import EndpointsDefinition


def test_endpoints_definition() -> None:
    config = ProcessingGraphConfig()
    graph = ProcessingGraph(config.specification)
    endpoint_config = EndpointConfig.from_env()

    endpoints_definition = EndpointsDefinition(graph, endpoint_config)
    assert endpoints_definition

    definition = endpoints_definition.steps_by_input_type_and_endpoint
    assert definition

    splits = definition["/splits"]
    assert splits is not None
    assert sorted(list(splits)) == ["config", "dataset"]
    assert splits["dataset"] is not None
    assert splits["config"] is not None
    assert len(splits["dataset"]) == 1  # Has one processing step
    assert len(splits["config"]) == 2  # Has two processing steps

    first_rows = definition["/first-rows"]
    assert first_rows is not None
    assert sorted(list(first_rows)) == ["split"]
    assert first_rows["split"] is not None
    assert len(first_rows["split"]) == 2  # Has two processing steps

    parquet = definition["/parquet"]
    assert parquet is not None
    assert sorted(list(parquet)) == ["config", "dataset"]
    assert parquet["dataset"] is not None
    assert parquet["config"] is not None
    assert len(parquet["dataset"]) == 1  # Only has one processing step
    assert len(parquet["config"]) == 1  # Only has one processing step

    dataset_info = definition["/info"]
    assert dataset_info is not None
    assert sorted(list(dataset_info)) == ["config", "dataset"]
    assert dataset_info["dataset"] is not None
    assert dataset_info["config"] is not None
    assert len(dataset_info["dataset"]) == 1  # Only has one processing step
    assert len(dataset_info["config"]) == 1  # Only has one processing step

    size = definition["/size"]
    assert size is not None
    assert sorted(list(size)) == ["config", "dataset"]
    assert size["dataset"] is not None
    assert size["config"] is not None
    assert len(size["dataset"]) == 1  # Only has one processing step
    assert len(size["config"]) == 1  # Only has one processing step

    opt_in_out_urls = definition["/opt-in-out-urls"]
    assert opt_in_out_urls is not None
    assert sorted(list(opt_in_out_urls)) == ["config", "dataset", "split"]
    assert opt_in_out_urls["split"] is not None
    assert opt_in_out_urls["config"] is not None
    assert opt_in_out_urls["dataset"] is not None
    assert len(opt_in_out_urls["split"]) == 1  # Only has one processing step
    assert len(opt_in_out_urls["config"]) == 1  # Only has one processing step
    assert len(opt_in_out_urls["dataset"]) == 1  # Only has one processing step

    # assert old deleted endpoints don't exist
    with raises(KeyError):
        _ = definition["/dataset-info"]
    with raises(KeyError):
        _ = definition["/parquet-and-dataset-info"]
    with raises(KeyError):
        _ = definition["/config-names"]
