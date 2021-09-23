import json
from typing import Optional, Union

from starlette.responses import Response

from datasets_preview_backend.types import Content


def to_bytes(content: Content) -> bytes:
    return json.dumps(
        content,
        ensure_ascii=False,
        allow_nan=False,
        indent=None,
        separators=(",", ":"),
    ).encode("utf-8")


class CustomJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: bytes) -> bytes:
        # content is already an UTF-8 encoded JSON
        return content


class CachedResponse:
    def __init__(self, content: Content, status_code: int = 200) -> None:
        # response content is encoded to avoid issues when caching ("/info" returns a non pickable object)
        self.content: Content = content
        self.status_code: int = status_code
        self.jsonContent: bytes = to_bytes(self.content)

    def is_error(self) -> bool:
        return self.status_code >= 300


def send(cached_response: CachedResponse, max_age: Optional[Union[int, None]] = None) -> Response:
    headers = {}
    if max_age is not None:
        headers["Cache-Control"] = f"public, max-age={max_age}"
    return CustomJSONResponse(cached_response.jsonContent, status_code=cached_response.status_code, headers=headers)
