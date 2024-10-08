from _typeshed import Incomplete
from collections.abc import Callable, Mapping
from re import Pattern
from typing import Protocol, TypeVar
from typing_extensions import ParamSpec

CUR_OS: str
IS_WIN: bool
IS_NIX: bool
RE_ANSI: Pattern[str]

class FormatReplace:
    replace: str
    format_called: int
    def __init__(self, replace: str = "") -> None: ...
    def __format__(self, _) -> str: ...

class _Has__Comparable(Protocol):
    _comparable: Incomplete

class Comparable:
    _comparable: Incomplete
    def __lt__(self, other: _Has__Comparable) -> bool: ...
    def __le__(self, other: _Has__Comparable) -> bool: ...
    def __eq__(self, other: _Has__Comparable) -> bool: ...  # type: ignore[override]
    def __ne__(self, other: _Has__Comparable) -> bool: ...  # type: ignore[override]
    def __gt__(self, other: _Has__Comparable) -> bool: ...
    def __ge__(self, other: _Has__Comparable) -> bool: ...

class ObjectWrapper:
    def __getattr__(self, name: str): ...
    def __setattr__(self, name: str, value) -> None: ...
    def wrapper_getattr(self, name): ...
    def wrapper_setattr(self, name, value): ...
    def __init__(self, wrapped) -> None: ...

class SimpleTextIOWrapper(ObjectWrapper):
    def __init__(self, wrapped, encoding) -> None: ...
    def write(self, s: str): ...
    def __eq__(self, other: object) -> bool: ...

_P = ParamSpec("_P")
_R = TypeVar("_R")

class DisableOnWriteError(ObjectWrapper):
    @staticmethod
    def disable_on_exception(tqdm_instance, func: Callable[_P, _R]) -> Callable[_P, _R]: ...
    def __init__(self, wrapped, tqdm_instance) -> None: ...
    def __eq__(self, other: object) -> bool: ...

class CallbackIOWrapper(ObjectWrapper):
    def __init__(self, callback: Callable[[int], object], stream, method: str = "read") -> None: ...

def disp_len(data: str) -> int: ...
def disp_trim(data: str, length: int) -> str: ...
def envwrap(
    prefix: str, types: Mapping[str, Callable[[Incomplete], Incomplete]] | None = None, is_method: bool = False
) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]: ...
