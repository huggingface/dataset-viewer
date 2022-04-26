from typing import Dict

from starlette.datastructures import Headers
from starlette.requests import Request

from api_service.middleware.token import get_token


def build_request(headers: Dict[str, str] = None) -> Request:
    if headers is None:
        headers = {}
    return Request({"type": "http", "headers": Headers(headers).raw})


def test_get_token() -> None:
    assert get_token(build_request({"Authorization": "Bearer some_token"})) == "some_token"
    assert get_token(build_request({"Authorization": "beArER some_token"})) == "some_token"
    assert get_token(build_request({"Authorization": "Basic some_token"})) is None
    assert get_token(build_request({"Authorization": "Bearersome_token"})) is None
    assert get_token(build_request({})) is None
