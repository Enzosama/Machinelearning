from collections.abc import Callable
from typing import TypeVar

_T = TypeVar("_T", bound=type)

# All these function actually mutate the given type:
def delegate_to_executor(*attrs: str) -> Callable[[_T], _T]: ...
def proxy_method_directly(*attrs: str) -> Callable[[_T], _T]: ...
def proxy_property_directly(*attrs: str) -> Callable[[_T], _T]: ...
def cond_delegate_to_executor(*attrs: str) -> Callable[[_T], _T]: ...
