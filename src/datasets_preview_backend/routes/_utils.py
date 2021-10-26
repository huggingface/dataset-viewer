import base64
from typing import Any

import orjson
from starlette.responses import JSONResponse, Response


# orjson is used to get rid of errors with datetime (see allenai/c4)
def orjson_default(obj: Any) -> Any:
    if isinstance(obj, bytes):
        return base64.b64encode(obj).decode("utf-8")
    raise TypeError


class OrjsonResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return orjson.dumps(content, option=orjson.OPT_UTC_Z, default=orjson_default)


def get_response(content: Any, status_code: int = 200, max_age: int = 0) -> Response:
    headers = {"Cache-Control": f"public, max-age={max_age}"} if max_age > 0 else {"Cache-Control": "no-store"}
    return OrjsonResponse(content, status_code=status_code, headers=headers)
