# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 The HuggingFace Authors.

from typing import Any, Callable, Coroutine

from starlette.requests import Request
from starlette.responses import Response

Endpoint = Callable[[Request], Coroutine[Any, Any, Response]]
