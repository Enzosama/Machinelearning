from typing import Any

from django.contrib.auth.backends import ModelBackend
from django.contrib.auth.base_user import AbstractBaseUser
from django.contrib.auth.models import AbstractUser, AnonymousUser
from django.db.models.options import Options
from django.http.request import HttpRequest
from django.test.client import Client

from .signals import user_logged_in as user_logged_in
from .signals import user_logged_out as user_logged_out
from .signals import user_login_failed as user_login_failed

SESSION_KEY: str
BACKEND_SESSION_KEY: str
HASH_SESSION_KEY: str
REDIRECT_FIELD_NAME: str

def load_backend(path: str) -> ModelBackend: ...
def get_backends() -> list[ModelBackend]: ...
def authenticate(
    request: HttpRequest | None = ..., **credentials: Any
) -> AbstractUser | None: ...
def login(
    request: HttpRequest,
    user: AbstractBaseUser | None,
    backend: type[ModelBackend] | str | None = ...,
) -> None: ...
def logout(request: HttpRequest) -> None: ...
async def aauthenticate(
    request: HttpRequest = ..., **credentials: Any
) -> AbstractBaseUser | None: ...
async def alogin(
    request: HttpRequest,
    user: AbstractBaseUser | None,
    backend: type[ModelBackend] | str | None = ...,
) -> None: ...
def get_user_model() -> type[AbstractUser]: ...
def get_user(request: HttpRequest | Client) -> AbstractUser | AnonymousUser: ...
def get_permission_codename(action: str, opts: Options[Any]) -> str: ...
def update_session_auth_hash(request: HttpRequest, user: AbstractBaseUser) -> None: ...

default_app_config: str
