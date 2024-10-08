from _typeshed import Incomplete

from networkx.utils.backends import _dispatch

@_dispatch
def full_rary_tree(r, n, create_using: Incomplete | None = None): ...
@_dispatch
def balanced_tree(r, h, create_using: Incomplete | None = None): ...
@_dispatch
def barbell_graph(m1, m2, create_using: Incomplete | None = None): ...
@_dispatch
def binomial_tree(n, create_using: Incomplete | None = None): ...
@_dispatch
def complete_graph(n, create_using: Incomplete | None = None): ...
@_dispatch
def circular_ladder_graph(n, create_using: Incomplete | None = None): ...
@_dispatch
def circulant_graph(n, offsets, create_using: Incomplete | None = None): ...
@_dispatch
def cycle_graph(n, create_using: Incomplete | None = None): ...
@_dispatch
def dorogovtsev_goltsev_mendes_graph(n, create_using: Incomplete | None = None): ...
@_dispatch
def empty_graph(n: Incomplete | int = 0, create_using: Incomplete | None = None, default=...): ...
@_dispatch
def ladder_graph(n, create_using: Incomplete | None = None): ...
@_dispatch
def lollipop_graph(m, n, create_using: Incomplete | None = None): ...
@_dispatch
def null_graph(create_using: Incomplete | None = None): ...
@_dispatch
def path_graph(n, create_using: Incomplete | None = None): ...
@_dispatch
def star_graph(n, create_using: Incomplete | None = None): ...
@_dispatch
def trivial_graph(create_using: Incomplete | None = None): ...
@_dispatch
def turan_graph(n, r): ...
@_dispatch
def wheel_graph(n, create_using: Incomplete | None = None): ...
@_dispatch
def complete_multipartite_graph(*subset_sizes): ...
