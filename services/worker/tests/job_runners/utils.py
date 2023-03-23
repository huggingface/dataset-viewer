# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

from http import HTTPStatus
from typing import Any, Mapping, Optional, TypedDict


class UpstreamResponse(TypedDict):
    kind: str
    dataset: str
    config: Optional[str]
    http_status: HTTPStatus
    content: Mapping[str, Any]
