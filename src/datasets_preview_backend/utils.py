from typing import Union

from starlette.requests import Request


def get_int_value(d, key, default):
    if key not in d:
        return default
    try:
        value = int(d.get(key))
    except (TypeError, ValueError):
        value = default
    return value


def get_str_value(d, key, default):
    if key not in d:
        return default
    value = str(d.get(key)).strip()
    return default if value == "" else value


def get_token(request: Request) -> Union[str, None]:
    try:
        if "Authorization" not in request.headers:
            return
        auth = request.headers["Authorization"]
        scheme, token = auth.split()
    except Exception:
        return
    if scheme.lower() != "bearer":
        return
    return token
