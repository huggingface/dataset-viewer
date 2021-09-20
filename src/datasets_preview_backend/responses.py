import json

from starlette.responses import Response

from datasets_preview_backend.types import ResponseContent, ResponseJSON


def to_bytes(content: ResponseContent) -> bytes:
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


class SerializedResponse:
    def __init__(self, content: ResponseContent, status_code: int = 200) -> None:
        # response content is encoded to avoid issues when caching ("/info" returns a non pickable object)
        self.content: bytes = to_bytes(content)
        self.status_code: int = status_code

    def as_json(self) -> ResponseJSON:
        return {
            "content": self.content,
            "status_code": self.status_code,
        }


def to_response(json: ResponseJSON) -> CustomJSONResponse:
    return CustomJSONResponse(json["content"], status_code=json["status_code"])
