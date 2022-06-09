import orjson
import base64
import pandas

from typing import Any


# orjson is used to get rid of errors with datetime (see allenai/c4), Timestamp (see ett)
def orjson_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    raise TypeError


def orjson_dumps(content: Any) -> bytes:
    return orjson.dumps(content, option=orjson.OPT_UTC_Z, default=orjson_default)
