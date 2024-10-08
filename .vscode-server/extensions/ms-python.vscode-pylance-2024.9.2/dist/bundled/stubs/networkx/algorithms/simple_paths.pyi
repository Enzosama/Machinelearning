from heapq import heappop, heappush
from itertools import count

from numpy.typing import ArrayLike

from ..classes.graph import Graph
from ..utils import not_implemented_for, pairwise

__all__ = [
    "all_simple_paths",
    "is_simple_path",
    "shortest_simple_paths",
    "all_simple_edge_paths",
]

def is_simple_path(G: Graph, nodes: ArrayLike) -> bool: ...
def all_simple_paths(G: Graph, source, target, cutoff=None): ...
def all_simple_edge_paths(G: Graph, source, target, cutoff=None): ...
def shortest_simple_paths(G: Graph, source, target, weight=None): ...

class PathBuffer:
    def __init__(self): ...
    def __len__(self): ...
    def push(self, cost, path): ...
    def pop(self): ...
