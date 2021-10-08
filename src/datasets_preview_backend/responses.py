from typing import Any

from starlette.responses import JSONResponse, Response

from datasets_preview_backend.exceptions import Status400Error, Status404Error
from datasets_preview_backend.types import Content


class CachedResponse:
    def __init__(self, content: Content, max_age: int, status_code: int = 200) -> None:
        # response content is encoded to avoid issues when caching ("/info" returns a non pickable object)
        self.content: Content = content
        self.status_code: int = status_code
        self.max_age: int = max_age

    def is_error(self) -> bool:
        return self.status_code >= 300

    def send(self) -> Response:
        headers = {"Cache-Control": f"public, max-age={self.max_age}"}
        return JSONResponse(self.content, status_code=self.status_code, headers=headers)


def get_cached_response(func: Any, max_age: int, **kwargs) -> CachedResponse:  # type: ignore
    try:
        content = func(**kwargs)
        return CachedResponse(content, max_age=max_age)
    except (Status400Error, Status404Error) as err:
        return CachedResponse(err.as_content(), status_code=err.status_code, max_age=max_age)
