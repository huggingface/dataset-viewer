# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from collections.abc import Callable, Coroutine
from typing import Any

from starlette.requests import Request
from starlette.responses import Response


def is_non_empty_string(string: Any) -> bool:
    return isinstance(string, str) and bool(string and string.strip())


def are_valid_parameters(parameters: list[Any]) -> bool:
    return all(is_non_empty_string(s) for s in parameters)


Endpoint = Callable[[Request], Coroutine[Any, Any, Response]]
