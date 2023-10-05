from starlette.requests import Request

from libapi.exceptions import InvalidParameterError


def get_request_parameter_offset(request: Request) -> int:
    offset = int(request.query_params.get("offset", 0))
    if offset < 0:
        raise InvalidParameterError(message="Parameter 'offset' must be positive")
    return offset
