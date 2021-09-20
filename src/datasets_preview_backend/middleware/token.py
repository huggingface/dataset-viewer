from typing import Tuple, Union

from starlette.authentication import AuthCredentials, AuthenticationBackend, BaseUser
from starlette.middleware import Middleware
from starlette.middleware.authentication import AuthenticationMiddleware
from starlette.requests import HTTPConnection

from datasets_preview_backend.constants import DEFAULT_DATASETS_ENABLE_PRIVATE


def get_token(request: HTTPConnection) -> Union[str, None]:
    try:
        if "Authorization" not in request.headers:
            return None
        auth = request.headers["Authorization"]
        scheme, token = auth.split()
    except Exception:
        return None
    if scheme.lower() != "bearer":
        return None
    return token


# it's not really correct: the token does not authenticate a user
class TokenUser(BaseUser):
    def __init__(self, token: str) -> None:
        self.username = "token"
        self._token = token

    @property
    def is_authenticated(self) -> bool:
        return True

    @property
    def display_name(self) -> str:
        return self.username

    @property
    def token(self) -> Union[str, None]:
        return self._token


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
    def __init__(self, datasets_enable_private: bool = DEFAULT_DATASETS_ENABLE_PRIVATE):
        super().__init__()
        self.datasets_enable_private = datasets_enable_private

    async def authenticate(
        self, request: HTTPConnection
    ) -> Tuple[AuthCredentials, Union[TokenUser, UnauthenticatedTokenUser]]:
        token = get_token(request)
        if token is None or not self.datasets_enable_private:
            return AuthCredentials([]), UnauthenticatedTokenUser()
        return AuthCredentials(["token"]), TokenUser(token)


def get_token_middleware(datasets_enable_private: bool = DEFAULT_DATASETS_ENABLE_PRIVATE) -> Middleware:
    return Middleware(
        AuthenticationMiddleware, backend=TokenAuthBackend(datasets_enable_private=datasets_enable_private)
    )
