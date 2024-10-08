from sympy.core.basic import Basic
from sympy.core.symbol import Dummy
from sympy.functions.elementary.piecewise import Piecewise

z = ...
timeit = ...

class _CoeffExpValueError(ValueError): ...

_dummies: dict[tuple[str, str], Dummy] = ...
_lookup_table = ...

def meijerint_indefinite(f, x) -> Piecewise | Basic | None: ...
@timeit
def meijerint_definite(f, x, a, b): ...
def meijerint_inversion(f, x, t) -> Piecewise | None: ...
