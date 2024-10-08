from sympy import Basic, Equality, Ne, Piecewise, Product
from sympy.core.relational import Relational
from sympy.matrices.expressions.matexpr import MatrixElement
from sympy.stats.joint_rv import JointDistribution, JointRandomSymbol
from sympy.stats.rv import RandomSymbol

__all__ = [
    "JointRV",
    "MultivariateNormal",
    "MultivariateLaplace",
    "Dirichlet",
    "GeneralizedMultivariateLogGamma",
    "GeneralizedMultivariateLogGammaOmega",
    "Multinomial",
    "MultivariateBeta",
    "MultivariateEwens",
    "MultivariateT",
    "NegativeMultinomial",
    "NormalGamma",
]

def multivariate_rv(cls, sym, *args) -> RandomSymbol | JointRandomSymbol: ...
def marginal_distribution(rv, *indices): ...

class JointDistributionHandmade(JointDistribution):
    _argnames = ...
    is_Continuous = ...
    @property
    def set(self) -> Basic: ...

def JointRV(symbol, pdf, _set=...) -> RandomSymbol | JointRandomSymbol: ...

class MultivariateNormalDistribution(JointDistribution):
    _argnames = ...
    is_Continuous = ...
    @property
    def set(self): ...
    @staticmethod
    def check(mu, sigma) -> None: ...
    def pdf(self, *args) -> MatrixElement: ...

def MultivariateNormal(name, mu, sigma) -> RandomSymbol | JointRandomSymbol: ...

class MultivariateLaplaceDistribution(JointDistribution):
    _argnames = ...
    is_Continuous = ...
    @property
    def set(self): ...
    @staticmethod
    def check(mu, sigma) -> None: ...
    def pdf(self, *args): ...

def MultivariateLaplace(name, mu, sigma) -> RandomSymbol | JointRandomSymbol: ...

class MultivariateTDistribution(JointDistribution):
    _argnames = ...
    is_Continuous = ...
    @property
    def set(self): ...
    @staticmethod
    def check(mu, sigma, v) -> None: ...
    def pdf(self, *args): ...

def MultivariateT(syms, mu, sigma, v) -> RandomSymbol | JointRandomSymbol: ...

class NormalGammaDistribution(JointDistribution):
    _argnames = ...
    is_Continuous = ...
    @staticmethod
    def check(mu, lamda, alpha, beta) -> None: ...
    @property
    def set(self): ...
    def pdf(self, x, tau): ...

def NormalGamma(sym, mu, lamda, alpha, beta) -> RandomSymbol | JointRandomSymbol: ...

class MultivariateBetaDistribution(JointDistribution):
    _argnames = ...
    is_Continuous = ...
    @staticmethod
    def check(alpha) -> None: ...
    @property
    def set(self): ...
    def pdf(self, *syms): ...

def MultivariateBeta(syms, *alpha) -> RandomSymbol | JointRandomSymbol: ...

Dirichlet = ...

class MultivariateEwensDistribution(JointDistribution):
    _argnames = ...
    is_Discrete = ...
    is_Continuous = ...
    @staticmethod
    def check(n, theta) -> None: ...
    @property
    def set(self) -> Equality | Relational | Ne | Product: ...
    def pdf(self, *syms) -> Piecewise: ...

def MultivariateEwens(syms, n, theta) -> RandomSymbol | JointRandomSymbol: ...

class GeneralizedMultivariateLogGammaDistribution(JointDistribution):
    _argnames = ...
    is_Continuous = ...
    def check(self, delta, v, l, mu) -> None: ...
    @property
    def set(self): ...
    def pdf(self, *y): ...

def GeneralizedMultivariateLogGamma(syms, delta, v, lamda, mu) -> RandomSymbol | JointRandomSymbol: ...
def GeneralizedMultivariateLogGammaOmega(syms, omega, v, lamda, mu) -> RandomSymbol | JointRandomSymbol: ...

class MultinomialDistribution(JointDistribution):
    _argnames = ...
    is_Continuous = ...
    is_Discrete = ...
    @staticmethod
    def check(n, p) -> None: ...
    @property
    def set(self): ...
    def pdf(self, *x) -> Piecewise: ...

def Multinomial(syms, n, *p) -> RandomSymbol | JointRandomSymbol: ...

class NegativeMultinomialDistribution(JointDistribution):
    _argnames = ...
    is_Continuous = ...
    is_Discrete = ...
    @staticmethod
    def check(k0, p) -> None: ...
    @property
    def set(self): ...
    def pdf(self, *k): ...

def NegativeMultinomial(syms, k0, *p) -> RandomSymbol | JointRandomSymbol: ...
