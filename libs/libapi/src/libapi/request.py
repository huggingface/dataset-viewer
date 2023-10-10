# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from libcommon.utils import MAX_NUM_ROWS_PER_PAGE
from starlette.requests import Request

from libapi.exceptions import InvalidParameterError


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
