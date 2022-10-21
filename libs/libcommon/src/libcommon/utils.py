# SPDX-License-Identifier: Apache-2.0
# Copyright 2022 The HuggingFace Authors.

import base64
from typing import Any

import orjson


# orjson is used to get rid of errors with datetime (see allenai/c4)
def orjson_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    raise TypeError


def orjson_dumps(content: Any) -> bytes:
    return orjson.dumps(content, option=orjson.OPT_UTC_Z, default=orjson_default)
