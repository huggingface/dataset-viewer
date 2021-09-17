import json
import typing

from starlette.responses import Response


def to_bytes(content: any) -> bytes:
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
    def __init__(self, content: typing.Any = None, status_code: int = 200) -> None:
        # response content is encoded to avoid issues when caching ("/info" returns a non pickable object)
        self.content = to_bytes(content)
        self.status_code = status_code

    def as_json(self):
        return {
            "content": self.content,
            "status_code": self.status_code,
        }


# TODO: create a type for the JSON
def to_response(json: typing.Any):
    return CustomJSONResponse(json["content"], status_code=json["status_code"])
