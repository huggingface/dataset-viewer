import json
from typing import Any, Dict, Optional

from starlette.responses import JSONResponse, Response

from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.types import Content


def to_bytes(content: Content) -> bytes:
    return json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
    ).encode("utf-8")


class CachedResponse:
    def __init__(self, content: Content, status_code: int = 200, max_age: Optional[int] = None) -> None:
        # response content is encoded to avoid issues when caching ("/info" returns a non pickable object)
        self.content: Content = content
        self.status_code: int = status_code
        self.max_age: Optional[int] = max_age

    def is_error(self) -> bool:
        return self.status_code >= 300

    def send(self) -> Response:
        headers = {}
        if self.max_age is not None:
            headers["Cache-Control"] = f"public, max-age={self.max_age}"
        return JSONResponse(self.content, status_code=self.status_code, headers=headers)


def get_cached_response(memoized_functions: Dict[str, Any], endpoint: str, **kwargs) -> CachedResponse:  # type: ignore
    try:
        content, max_age = memoized_functions[endpoint](**kwargs, _return_max_age=True)
        return CachedResponse(content, max_age=max_age)
    except (Status400Error, Status404Error) as err:
        return CachedResponse(err.as_content(), status_code=err.status_code)
