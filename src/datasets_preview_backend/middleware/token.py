from typing import Union

from starlette.authentication import AuthCredentials, AuthenticationBackend, BaseUser
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import Request


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


# it's not really correct: the token does not authenticate a user
class TokenUser(BaseUser):
    def __init__(self, token: str) -> None:
        self.username = "token"
        self.token = token

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def display_name(self) -> str:
        return self.username

    @property
    def token(self) -> Union[str, None]:
        return self.token


class UnauthenticatedTokenUser(BaseUser):
    @property
    def is_authenticated(self) -> bool:
        return False

    @property
    def display_name(self) -> str:
        return ""

    @property
    def token(self) -> Union[str, None]:
        return None


class TokenAuthBackend(AuthenticationBackend):
    async def authenticate(self, request):
        token = get_token(request)
        if token is None:
            return AuthCredentials([]), UnauthenticatedTokenUser()
        return AuthCredentials(["token"]), TokenUser(token)


def get_token_middleware():
    return Middleware(AuthenticationMiddleware, backend=TokenAuthBackend())
