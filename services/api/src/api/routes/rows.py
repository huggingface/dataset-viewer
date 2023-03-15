# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import logging
from functools import lru_cache, partial
from typing import Any, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from hffs.fs import HfFileSystem
from libcommon.processing_graph import ProcessingStep
from starlette.requests import Request
from starlette.responses import Response
from tqdm.contrib.concurrent import thread_map
from typing_extensions import Protocol

from api.authentication import auth_check
from api.prometheus import StepProfiler
from api.routes.endpoint import get_cache_entry_from_steps
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

MAX_ROWS = 100

PARQUET_REVISION = "refs/convert/parquet"

# TODO: manage private/gated datasets


class FileSystemError(Exception):
    pass


class ParquetResponseFormatError(Exception):
    pass


class ParquetResponseEmptyError(Exception):
    pass


# TODO: how to invalidate the cache when the parquet branch is created or deleted?
@lru_cache(maxsize=128)
def get_parquet_fs(dataset: str) -> HfFileSystem:
    """Get the parquet filesystem for a dataset.

    The parquet files are stored in a separate branch of the dataset repository (see PARQUET_REVISION)

    Args:
        dataset (str): The dataset name.

    Returns:
        HfFileSystem: The parquet filesystem.
    """
    return HfFileSystem(dataset, repo_type="dataset", revision=PARQUET_REVISION)


# RowGroupReaderBase = Callable[[], pa.Table]
# RowGroupReader = Union[RowGroupReaderBase, partial[RowGroupReaderBase]]
RowGroupReader = partial[Any]

Index = Tuple[Any, List[RowGroupReader]]


class Indexer(Protocol):
    def __call__(self, dataset: str, config: str, split: str) -> Index:
        ...


def create_index(
    config_parquet_processing_steps: List[ProcessingStep],
    init_processing_steps: List[ProcessingStep],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
) -> Indexer:
    # _lru_cache_wrapper
    @lru_cache(maxsize=128)
    def index(
        dataset: str,
        config: str,
        split: str,
    ) -> Index:
        with StepProfiler(method="rows.index", step="all"):
            # get the list of parquet files
            with StepProfiler(method="rows.index", step="get list of parquet files for split"):
                try:
                    result = get_cache_entry_from_steps(
                        processing_steps=config_parquet_processing_steps,
                        dataset=dataset,
                        config=config,
                        split=None,
                        init_processing_steps=init_processing_steps,
                        hf_endpoint=hf_endpoint,
                        hf_token=hf_token,
                    )
                    content = result["content"]
                except ApiCustomError as e:
                    raise e
                except Exception as e:
                    raise UnexpectedError("Could not get the list of parquet files to fetch the rows from.") from e
                    # ^ TODO: improve the error, depending on the case
                try:
                    sources = sorted(
                        f"{config}/{parquet_file['filename']}"
                        for parquet_file in content["parquet_files"]
                        if parquet_file["split"] == split and parquet_file["config"] == config
                    )
                except Exception as e:
                    raise ParquetResponseFormatError(f"Could not parse the list of parquet files: {e}") from e
                logging.debug(f"Found {len(sources)} parquet files for split {split}: {sources}")
                if not sources:
                    raise ParquetResponseEmptyError("No parquet files found.")
            with StepProfiler(method="rows.index", step="get the Hub's dataset filesystem"):
                fs = get_parquet_fs(dataset)
            with StepProfiler(method="rows.index", step="get one parquet reader per parquet file"):
                desc = f"{dataset}/{config}/{split}"
                try:
                    parquet_files: List[pq.ParquetFile] = thread_map(
                        partial(pq.ParquetFile, filesystem=fs),
                        sources,
                        desc=desc,
                        unit="pq",
                        tqdm_class=None,
                    )
                except Exception as e:
                    raise FileSystemError(f"Could not read the parquet files: {e}") from e
            # features = Features.from_arrow_schema(all_pf[0].schema.to_arrow_schema())
            # columns = [
            #     col
            #     for col in features
            #     if all(bad_type not in str(features[col]) for bad_type in ["Image(", "Audio(", "'binary'"])
            # ]
            columns = None
            # info = (
            #     ""
            #     if len(columns) == len(features)
            #     else f"Some columns are not supported yet: {sorted(set(features) - set(columns))}"
            # )
            with StepProfiler(method="rows.index", step="create the row group offsets"):
                row_group_offsets = np.cumsum(
                    [
                        parquet_file.metadata.row_group(group_id).num_rows
                        for parquet_file in parquet_files
                        for group_id in range(parquet_file.metadata.num_row_groups)
                    ]
                )
            with StepProfiler(method="rows.index", step="create the row group readers"):
                row_group_readers = [
                    partial(parquet_file.read_row_group, i=group_id, columns=columns)
                    for parquet_file in parquet_files
                    for group_id in range(parquet_file.metadata.num_row_groups)
                ]
            return row_group_offsets, row_group_readers

    return index


def query(offset: int, length: int, row_group_offsets: Any, row_group_readers: List[RowGroupReader]) -> pa.Table:
    """Query the parquet files

    Note that this implementation will always read one row group, to get the list of columns and always have the same
    schema, even if the requested rows are invalid (out of range).

    Args:
        offset (int): The first row to read.
        length (int): The number of rows to read.
        row_group_offsets (Any): The row group offsets. See index().
        row_group_readers (List[RowGroupReader]): The row group readers. See index().

    Returns:
        pa.Table: The requested rows.
    """
    if (len(row_group_offsets) == 0) or (len(row_group_readers) == 0):
        raise ParquetResponseEmptyError("No parquet files found.")
    last_row_in_parquet = row_group_offsets[-1] - 1
    first_row = min(offset, last_row_in_parquet)
    last_row = min(offset, offset + length - 1, last_row_in_parquet)
    first_row_group_id, last_row_group_id = np.searchsorted(row_group_offsets, [first_row, last_row], side="right")
    pa_table = pa.concat_tables([row_group_readers[i]() for i in range(first_row_group_id, last_row_group_id + 1)])
    first_row_in_pa_table = row_group_offsets[first_row_group_id - 1] if first_row_group_id > 0 else 0
    return pa_table.slice(offset - first_row_in_pa_table, length)


def create_rows_endpoint(
    config_parquet_processing_steps: List[ProcessingStep],
    init_processing_steps: List[ProcessingStep],
    hf_endpoint: str,
    hf_token: Optional[str] = None,
    hf_jwt_public_key: Optional[str] = None,
    hf_jwt_algorithm: Optional[str] = None,
    external_auth_url: Optional[str] = None,
    hf_timeout_seconds: Optional[float] = None,
    max_age_long: int = 0,
    max_age_short: int = 0,
) -> Endpoint:
    index = create_index(
        config_parquet_processing_steps=config_parquet_processing_steps,
        init_processing_steps=init_processing_steps,
        hf_endpoint=hf_endpoint,
        hf_token=hf_token,
    )

    async def rows_endpoint(request: Request) -> Response:
        with StepProfiler(method="rows_endpoint", step="all"):
            try:
                with StepProfiler(method="rows_endpoint", step="validate parameters"):
                    dataset = request.query_params.get("dataset")
                    config = request.query_params.get("config")
                    split = request.query_params.get("split")
                    if not dataset or not config or not split or not are_valid_parameters([dataset, config, split]):
                        raise MissingRequiredParameterError("Parameter 'dataset', 'config' and 'split' are required")
                    offset = int(request.query_params.get("offset", 0))
                    if offset < 0:
                        raise InvalidParameterError(message="Offset must be positive")
                    length = int(request.query_params.get("length", MAX_ROWS))
                    if length < 0:
                        raise InvalidParameterError("Length must be positive")
                    if length > MAX_ROWS:
                        raise InvalidParameterError(f"Length must be less than or equal to {MAX_ROWS}")
                    logging.info(
                        f"/rows, dataset={dataset}, config={config}, split={split}, offset={offset}, length={length}"
                    )

                with StepProfiler(method="rows_endpoint", step="check authentication"):
                    # if auth_check fails, it will raise an exception that will be caught below
                    auth_check(
                        dataset,
                        external_auth_url=external_auth_url,
                        request=request,
                        hf_jwt_public_key=hf_jwt_public_key,
                        hf_jwt_algorithm=hf_jwt_algorithm,
                        hf_timeout_seconds=hf_timeout_seconds,
                    )

                with StepProfiler(method="rows_endpoint", step="get row groups index"):
                    row_group_offsets, row_group_readers = index(dataset=dataset, config=config, split=split)
                with StepProfiler(method="rows_endpoint", step="query the rows"):
                    pa_table = query(
                        offset=offset,
                        length=length,
                        row_group_offsets=row_group_offsets,
                        row_group_readers=row_group_readers,
                    )
                with StepProfiler(method="rows_endpoint", step="transform to a list"):
                    # TODO: ignore, or transform, some of the cells (e.g. audio or image)
                    rows = pa_table.to_pylist()
                    # TODO: add features?
                with StepProfiler(method="rows_endpoint", step="generate the OK response"):
                    return get_json_ok_response(content={"rows": rows}, max_age=max_age_long)
            except Exception as e:
                error = e if isinstance(e, ApiCustomError) else UnexpectedError("Unexpected error.", e)
                with StepProfiler(method="rows_endpoint", step="generate API error response"):
                    return get_json_api_error_response(error=error, max_age=max_age_short)

    return rows_endpoint
