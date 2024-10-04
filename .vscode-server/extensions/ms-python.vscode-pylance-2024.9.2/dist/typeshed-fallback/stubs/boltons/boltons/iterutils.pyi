from _typeshed import Incomplete
from collections.abc import Generator

def is_iterable(obj) -> bool: ...
def is_scalar(obj) -> bool: ...
def is_collection(obj) -> bool: ...
def split(src, sep: Incomplete | None = None, maxsplit: Incomplete | None = None): ...
def split_iter(
    src, sep: Incomplete | None = None, maxsplit: Incomplete | None = None
) -> Generator[Incomplete, None, Incomplete]: ...
def lstrip(iterable, strip_value: Incomplete | None = None): ...
def lstrip_iter(iterable, strip_value: Incomplete | None = None) -> Generator[Incomplete, None, None]: ...
def rstrip(iterable, strip_value: Incomplete | None = None): ...
def rstrip_iter(iterable, strip_value: Incomplete | None = None) -> Generator[Incomplete, None, None]: ...
def strip(iterable, strip_value: Incomplete | None = None): ...
def strip_iter(iterable, strip_value: Incomplete | None = None): ...
def chunked(src, size, count: Incomplete | None = None, **kw): ...
def chunked_iter(src, size, **kw) -> Generator[Incomplete, None, Incomplete]: ...
def chunk_ranges(
    input_size: int, chunk_size: int, input_offset: int = 0, overlap_size: int = 0, align: bool = False
) -> Generator[tuple[int, int], None, None]: ...
def pairwise(src, end=...): ...
def pairwise_iter(src, end=...): ...
def windowed(src, size, fill=...): ...
def windowed_iter(src, size, fill=...): ...
def xfrange(stop, start: Incomplete | None = None, step: float = 1.0) -> Generator[Incomplete, None, None]: ...
def frange(stop, start: Incomplete | None = None, step: float = 1.0): ...
def backoff(start, stop, count: Incomplete | None = None, factor: float = 2.0, jitter: bool = False): ...
def backoff_iter(
    start, stop, count: Incomplete | None = None, factor: float = 2.0, jitter: bool = False
) -> Generator[Incomplete, None, None]: ...
def bucketize(src, key=..., value_transform: Incomplete | None = None, key_filter: Incomplete | None = None): ...
def partition(src, key=...): ...
def unique(src, key: Incomplete | None = None): ...
def unique_iter(src, key: Incomplete | None = None) -> Generator[Incomplete, None, Incomplete]: ...
def redundant(src, key: Incomplete | None = None, groups: bool = False): ...
def one(src, default: Incomplete | None = None, key: Incomplete | None = None): ...
def first(iterable, default: Incomplete | None = None, key: Incomplete | None = None): ...
def flatten_iter(iterable) -> Generator[Incomplete, None, None]: ...
def flatten(iterable): ...
def same(iterable, ref=...): ...
def default_visit(path, key, value): ...
def default_enter(path, key, value): ...
def default_exit(path, key, old_parent, new_parent, new_items): ...
def remap(root, visit=..., enter=..., exit=..., **kwargs): ...

class PathAccessError(KeyError, IndexError, TypeError):
    exc: Incomplete
    seg: Incomplete
    path: Incomplete
    def __init__(self, exc, seg, path) -> None: ...

def get_path(root, path, default=...): ...
def research(root, query=..., reraise: bool = False): ...

class GUIDerator:
    size: Incomplete
    count: Incomplete
    def __init__(self, size: int = 24) -> None: ...
    pid: Incomplete
    salt: Incomplete
    def reseed(self) -> None: ...
    def __iter__(self): ...
    def __next__(self): ...
    next: Incomplete

class SequentialGUIDerator(GUIDerator):
    start: Incomplete
    def reseed(self) -> None: ...
    def __next__(self): ...
    next: Incomplete

guid_iter: Incomplete
seq_guid_iter: Incomplete

def soft_sorted(
    iterable,
    first: Incomplete | None = None,
    last: Incomplete | None = None,
    key: Incomplete | None = None,
    reverse: bool = False,
): ...
def untyped_sorted(iterable, key: Incomplete | None = None, reverse: bool = False): ...
