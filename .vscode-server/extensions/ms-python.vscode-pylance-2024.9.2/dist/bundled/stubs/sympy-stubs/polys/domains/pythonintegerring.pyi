from typing import Any, Literal

from sympy.core.numbers import Integer
from sympy.polys.domains.groundtypes import PythonInteger
from sympy.polys.domains.integerring import IntegerRing
from sympy.utilities import public

class PythonIntegerRing(IntegerRing):
    dtype = PythonInteger
    zero = dtype(0)
    one = dtype(1)
    alias = ...
    def __init__(self) -> None: ...
    def to_sympy(self, a) -> Integer: ...
    def from_sympy(self, a) -> PythonInteger: ...
    def from_FF_python(K1, a, K0): ...
    def from_ZZ_python(K1, a, K0): ...
    def from_QQ(K1, a, K0) -> None: ...
    def from_QQ_python(K1, a, K0) -> None: ...
    def from_FF_gmpy(K1, a, K0) -> PythonInteger: ...
    def from_ZZ_gmpy(K1, a, K0) -> PythonInteger: ...
    def from_QQ_gmpy(K1, a, K0) -> PythonInteger | None: ...
    def from_RealField(K1, a, K0) -> PythonInteger | None: ...
    def gcdex(
        self, a, b
    ) -> (
        tuple[Literal[0], Literal[1], Literal[0]]
        | tuple[Literal[0], Any, Any]
        | tuple[Any, Literal[0], Any]
        | tuple[Literal[-1, 1, 0], Literal[0, -1, 1], Any]
    ): ...
    def gcd(self, a, b) -> int: ...
    def lcm(self, a, b) -> Literal[0]: ...
    def sqrt(self, a) -> int: ...
    def factorial(self, a) -> int: ...
