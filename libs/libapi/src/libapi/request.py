# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.
from libcommon.utils import MAX_NUM_ROWS_PER_PAGE
from starlette.requests import Request

from libapi.exceptions import InvalidParameterError, MissingRequiredParameterError
from libapi.utils import are_valid_parameters


def get_request_parameter_length(request: Request) -> int:
    try:
        length = int(request.query_params.get("length", MAX_NUM_ROWS_PER_PAGE))
    except ValueError:
        raise InvalidParameterError("Parameter 'length' must be integer")
    if length < 0:
        raise InvalidParameterError("Parameter 'length' must be positive")
    elif length > MAX_NUM_ROWS_PER_PAGE:
        raise InvalidParameterError(f"Parameter 'length' must not be greater than {MAX_NUM_ROWS_PER_PAGE}")
    return length


def get_request_parameter_offset(request: Request) -> int:
    try:
        offset = int(request.query_params.get("offset", 0))
    except ValueError:
        raise InvalidParameterError("Parameter 'offset' must be integer")
    if offset < 0:
        raise InvalidParameterError(message="Parameter 'offset' must be positive")
    return offset


def get_request_parameter_dataset(request: Request) -> str:
    dataset = request.query_params.get("dataset")
    if not dataset or not are_valid_parameters([dataset]):
        raise MissingRequiredParameterError("Parameter 'dataset' is required")
    return dataset


def get_request_parameter_config(request: Request) -> str:
    config = request.query_params.get("config")
    if not config or not are_valid_parameters([config]):
        raise MissingRequiredParameterError("Parameter 'config' is required")
    return config


def get_request_parameter_split(request: Request) -> str:
    split = request.query_params.get("split")
    if not split or not are_valid_parameters([split]):
        raise MissingRequiredParameterError("Parameter 'split' is required")
    return split
