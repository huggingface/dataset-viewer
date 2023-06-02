# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from os import PathLike
from typing import List, Optional, Set, Union

import duckdb
from libcommon.processing_graph import ProcessingGraph
from libcommon.prometheus import StepProfiler
from libcommon.simple_cache import get_valid_datasets
from starlette.requests import Request
from starlette.responses import Response

from api.routes.endpoint import get_cache_entry_from_steps
from api.utils import (
    Endpoint,
    MissingRequiredParameterError,
    UnexpectedError,
    are_valid_parameters,
    get_json_api_error_response,
    get_json_ok_response,
)


def get_valid(processing_graph: ProcessingGraph) -> List[str]:
    # a dataset is considered valid if at least one response for PROCESSING_STEPS_FOR_VALID
    # is valid.
    datasets: Optional[Set[str]] = None
    for processing_step in processing_graph.get_processing_steps_required_by_dataset_viewer():
        kind_datasets = get_valid_datasets(kind=processing_step.cache_kind)
        if datasets is None:
            # first iteration fills the set of datasets
            datasets = kind_datasets
        else:
            # next iterations remove the datasets that miss a required processing step
            datasets.intersection_update(kind_datasets)
    # note that the list is sorted alphabetically for consistency
    return [] if datasets is None else sorted(datasets)


StrPath = Union[str, PathLike[str]]


def create_fts_endpoint(
    processing_graph: ProcessingGraph,
    cached_assets_directory: StrPath,
    hf_endpoint: str,
    max_age_long: int = 0,
    max_age_short: int = 0,
    hf_token: Optional[str] = None,
) -> Endpoint:
    async def fts_endpoint(request: Request) -> Response:
        with StepProfiler(method="fts_endpoint", step="all"):
            try:
                logging.info("/fts")
                # processing_step = processing_graph.get_processing_step("split-duckdb-index")
                dataset = request.query_params.get("dataset")
                config = request.query_params.get("config")
                split = request.query_params.get("split")
                query = request.query_params.get("query")
                if not dataset or not config or not split or not are_valid_parameters([dataset, config, split]):
                    raise MissingRequiredParameterError("Parameter 'dataset', 'config' and 'split' are required")
                if not query:
                    raise MissingRequiredParameterError("Parameter 'query' is required")
                # upstream_result = get_cache_entry_from_steps(
                #     processing_steps=[processing_step],
                #     dataset=dataset,
                #     config=config,
                #     split=split,
                #     processing_graph=processing_graph,
                #     hf_endpoint=hf_endpoint,
                #     hf_token=hf_token,
                # )
                # content = result["content"]
                # duck_db_name = content["duckdb_db_name"]

            except Exception as e:
                with StepProfiler(method="fts_endpoint", step="generate API error response"):
                    return get_json_api_error_response(UnexpectedError("Unexpected error.", e), max_age=max_age_short)
            duckdb.execute("INSTALL 'fts';")
            duckdb.execute("LOAD 'fts';")
            # db_location = cached_assets_directory / duck_db_name
            db_location = "/tmp/asoria/openfire/--/default/train/index.db"
            con = duckdb.connect(str(db_location))
            result = con.execute(
                (
                    "SELECT fts_main_data.match_bm25(id, ?) AS score, * FROM data WHERE score IS NOT NULL ORDER BY"
                    " score DESC;"
                ),
                [query],
            ).df()
            return get_json_ok_response({"result": result.to_json()}, max_age=max_age_long)

    return fts_endpoint
