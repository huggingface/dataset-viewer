# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
import os.path
from typing import Any, Mapping, Optional, TypedDict

import duckdb
import pyarrow.parquet as pq
from datasets import Features
from libcommon.parquet_utils import (
    ParquetFileMetadataItem,
    StrPath,
    get_supported_unsupported_columns,
)
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import get_previous_step_or_raise
from libcommon.viewer_utils.features import get_cell_value
from starlette.requests import Request
from starlette.responses import Response

from api.authentication import auth_check
from api.utils import (
    ApiCustomError,
    Endpoint,
    InvalidParameterError,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_ok_response,
)

# TODO: duplicated in /rows
MAX_ROWS = 100

# TODO: duplicated in /rows
# audio still has some errors when librosa is imported
UNSUPPORTED_FEATURES_MAGIC_STRINGS = ["'binary'", "Audio("]


Feature = Mapping[str, Any]


# TODO: duplicated
class FeatureItem(TypedDict):
    feature_idx: int
    name: str
    type: Feature


# TODO: duplicated
Row = Mapping[str, Any]


# TODO: duplicated
class RowItem(TypedDict):
    row_idx: int
    row: Mapping[str, Any]
    truncated_cells: list[str]


class Table(TypedDict):
    columns: list[str]
    rows: list[tuple[Any, ...]]


logger = logging.getLogger(__name__)

# DuckDB connection
con = duckdb.connect()
con.execute("INSTALL httpfs;")
con.execute("LOAD httpfs;")


def create_filter_endpoint(
    processing_graph: ProcessingGraph,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    parquet_metadata_directory: StrPath,
    hf_jwt_public_key: Optional[str] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    async def filter_endpoint(request: Request) -> Response:
        revision: Optional[str] = None
        with StepProfiler(method="filter_endpoint", step="all"):
            try:
                with StepProfiler(method="filter_endpoint", step="validate parameters"):
                    dataset = request.query_params.get("dataset")
                    config = request.query_params.get("config")
                    split = request.query_params.get("split")
                    if not dataset or not config or not split or not are_valid_parameters([dataset, config, split]):
                        raise MissingRequiredParameterError("Parameter 'dataset', 'config' and 'split' are required")
                    # TODO: validate where
                    where = request.query_params.get("where")
                    if not where:
                        raise MissingRequiredParameterError("Parameter 'where' is required")
                    offset = int(request.query_params.get("offset", 0))
                    if offset < 0:
                        raise InvalidParameterError(message="Offset must be positive")
                    length = int(request.query_params.get("length", MAX_ROWS))
                    if length < 0:
                        raise InvalidParameterError("Length must be positive")
                    if length > MAX_ROWS:
                        raise InvalidParameterError(f"Length must be less than or equal to {MAX_ROWS}")
                    logger.info(
                        f'/filter, dataset={dataset}, config={config}, split={split}, where="{where}",'
                        f" offset={offset}, length={length}"
                    )
                with StepProfiler(method="filter_endpoint", step="check authentication"):
                    # If auth_check fails, it will raise an exception that will be caught below
                    auth_check(
                        dataset=dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_key=hf_jwt_public_key,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )
                with StepProfiler(method="filter_endpoint", step="get parquet file metadata items from cache"):
                    parquet_file_metadata_items, revision = get_config_parquet_metadata_from_cache(
                        dataset=dataset, config=config, split=split, processing_graph=processing_graph
                    )
                with StepProfiler(method="filter_endpoint", step="get parquet file urls"):
                    parquet_file_urls = [item["url"] for item in parquet_file_metadata_items]
                with StepProfiler(method="filter_endpoint", step="get features"):
                    features = get_features_from_parquet_file_metadata(
                        parquet_file_metadata_item=parquet_file_metadata_items[0],
                        parquet_metadata_directory=parquet_metadata_directory,
                    )
                with StepProfiler(method="filter_endpoint", step="get supported and unsupported columns"):
                    supported_columns, _ = get_supported_unsupported_columns(
                        features,
                        unsupported_features_magic_strings=UNSUPPORTED_FEATURES_MAGIC_STRINGS,
                    )
                with StepProfiler(method="filter_endpoint", step="execute filter query"):
                    table = execute_filter_query(
                        columns=supported_columns,
                        parquet_file_urls=parquet_file_urls,
                        where=where,
                        limit=length,
                        offset=offset,
                    )
                with StepProfiler(method="filter_endpoint", step="create response"):
                    response = create_response(
                        dataset=dataset,
                        config=config,
                        split=split,
                        cached_assets_base_url=cached_assets_base_url,
                        cached_assets_directory=cached_assets_directory,
                        table=table,
                        offset=offset,
                        features=features,
                    )
                with StepProfiler(method="filter_endpoint", step="generate the OK response"):
                    return get_json_ok_response(content=response, max_age=max_age_long, revision=revision)
            except Exception as e:
                error = e if isinstance(e, ApiCustomError) else UnexpectedError("Unexpected error.", e)
                with StepProfiler(method="filter_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short, revision=revision)

    return filter_endpoint


def get_config_parquet_metadata_from_cache(
    dataset: str, config: str, split: str, processing_graph: ProcessingGraph
) -> tuple[list[ParquetFileMetadataItem], Optional[str]]:
    config_parquet_metadata_processing_steps = processing_graph.get_config_parquet_metadata_processing_steps()
    if not config_parquet_metadata_processing_steps:
        raise RuntimeError("No processing steps are configured to provide config-parquet-metadata response.")
    cache_kinds = [step.cache_kind for step in config_parquet_metadata_processing_steps]
    try:
        result = get_previous_step_or_raise(
            kinds=cache_kinds,
            dataset=dataset,
            config=config,
        )
    except Exception as e:
        raise UnexpectedError("Could not get the list of parquet files metadata.") from e
    response = result.response
    revision = response["dataset_git_revision"]
    parquet_file_metadata_items = response["content"]["parquet_files_metadata"]
    parquet_file_metadata_items = [
        item for item in parquet_file_metadata_items if item["split"] == split and item["config"] == config
    ]
    return parquet_file_metadata_items, revision


def get_features_from_parquet_file_metadata(
    parquet_file_metadata_item: ParquetFileMetadataItem, parquet_metadata_directory: StrPath
) -> Features:
    parquet_file_metadata_path = os.path.join(
        parquet_metadata_directory, parquet_file_metadata_item["parquet_metadata_subpath"]
    )
    return Features.from_arrow_schema(pq.read_schema(parquet_file_metadata_path))


def execute_filter_query(
    columns: list[str], parquet_file_urls: list[str], where: str, limit: int, offset: int
) -> Table:
    # TODO: Address possible SQL injection CWE-89
    query = con.sql(
        f"""\
        SELECT {",".join(columns)}
        FROM read_parquet({parquet_file_urls})
        WHERE {where}
        LIMIT {limit}
        OFFSET {offset}"""  # nosec B608
    )
    rows = query.fetchall()
    return {"columns": columns, "rows": rows}


# TODO: duplicated in /rows except Table
def create_response(
    dataset: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    table: Table,
    offset: int,
    features: Features,
) -> Any:
    return {
        "features": to_features_list(features),
        "rows": to_rows_list(
            table,
            dataset,
            config,
            split,
            cached_assets_base_url,
            cached_assets_directory,
            offset,
            features,
        ),
    }


# TODO: duplicated in /rows
def to_features_list(features: Features) -> list[FeatureItem]:
    features_dict = features.to_dict()
    return [
        {
            "feature_idx": idx,
            "name": name,
            "type": features_dict[name],
        }
        for idx, name in enumerate(features)
    ]


# TODO: duplicated in /rows except Table
def to_rows_list(
    table: Table,
    dataset: str,
    config: str,
    split: str,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    offset: int,
    features: Features,
) -> list[RowItem]:
    # transform the rows, if needed (e.g. save the images or audio to the assets, and return their URL)
    transformed_rows = transform_rows(
        dataset=dataset,
        config=config,
        split=split,
        table=table,
        features=features,
        cached_assets_base_url=cached_assets_base_url,
        cached_assets_directory=cached_assets_directory,
        offset=offset,
    )
    return [
        {
            "row_idx": idx + offset,
            "row": row,
            "truncated_cells": [],
        }
        for idx, row in enumerate(transformed_rows)
    ]


# TODO: duplicated in /rows except Table
def transform_rows(
    dataset: str,
    config: str,
    split: str,
    table: Table,
    features: Features,
    cached_assets_base_url: str,
    cached_assets_directory: StrPath,
    offset: int,
) -> list[Row]:
    return [
        {
            featureName: get_cell_value(
                dataset=dataset,
                config=config,
                split=split,
                row_idx=offset + row_idx,
                cell=row[table["columns"].index(featureName)] if featureName in table["columns"] else None,
                featureName=featureName,
                fieldType=fieldType,
                assets_base_url=cached_assets_base_url,
                assets_directory=cached_assets_directory,
            )
            for (featureName, fieldType) in features.items()
        }
        for row_idx, row in enumerate(table["rows"])
    ]
