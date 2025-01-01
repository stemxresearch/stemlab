from numpy import angle, euler_gamma, exp, log, pi, sqrt, e
from scipy.special import gamma

class MathConstants:
    """
    A class to encapsulate common mathematical const.

    For more details, see the list of mathematical constants on Wikipedia:
    https://en.wikipedia.org/wiki/List_of_mathematical_constants
    
    """
    @property
    def pi(self) -> float:
        """Returns the mathematical constant pi."""
        return pi

    @property
    def sqrt2(self) -> float:
        """Returns the square root of 2."""
        return sqrt(2)

    @property
    def sqrt3(self) -> float:
        """Returns the square root of 3."""
        return sqrt(3)

    @property
    def sqrt5(self) -> float:
        """Returns the square root of 5."""
        return sqrt(5)

    @property
    def cuberoot_of_root2(self) -> float:
        """Returns the cube root of root 2."""
        return sqrt(2) ** (1 / 3)

    @property
    def cuberoot_of_root3(self) -> float:
        """Returns the cube root of root 3."""
        return sqrt(3) ** (1 / 3)

    @property
    def phi(self) -> float:
        """Returns the golden ratio."""
        return (1 + sqrt(5)) / 2

    @property
    def tau(self) -> float:
        """Returns the value of 2 * pi."""
        return 2 * pi

    @property
    def psi(self) -> float:
        """Returns the supergolden ratio."""
        return (1 + ((29 + 3 * sqrt(93)) / 2) ** (1 / 3) +
                ((29 - 3 * sqrt(93)) / 2) ** (1 / 3)) / 3

    @property
    def r12th_root_of_root2(self) -> float:
        """Returns the 12th root of root 2."""
        return sqrt(2) ** (1 / 12)

    @property
    def mu(self) -> float:
        """Returns the connective constant."""
        return sqrt(2 + sqrt(2))

    @property
    def e(self) -> float:
        """Returns the base of natural logarithms."""
        return e

    @property
    def ln2(self) -> float:
        """Returns the natural logarithm of 2."""
        return log(2)

    @property
    def euler_gamma(self) -> float:
        """Returns the Euler-Gamma constant."""
        return euler_gamma

    @property
    def omega(self) -> float:
        """Returns the omega constant."""
        return 0.56714329040978387299

    @property
    def exp_pi(self) -> float:
        """Returns the value of e raised to the power of pi."""
        return exp(pi)

    @property
    def root2_to_root2(self) -> float:
        """Returns the value of square root of 2 raised to the power of sqrt(2)."""
        return sqrt(2) ** sqrt(2)

    @property
    def euler_gompertz_delta(self) -> float:
        """Returns the Euler-Gompertz constant."""
        return 0.59634736232319407434

    @property
    def levy_beta(self) -> float:
        """Returns the Euler-Gompertz constant."""
        return pi ** 2 / (12 * log(2))

    @property
    def levy_ebeta(self) -> float:
        """Returns the Euler-Gompertz constant."""
        return exp(pi ** 2 / (12 * log(2)))

    @property
    def siver_ratio_delta(self) -> float:
        """Returns the Silver ratio."""
        return sqrt(2) + 1

    @property
    def second_hermite_gamma(self) -> float:
        """Returns the second Hermite constant."""
        return 2 / sqrt(3)

    @property
    def gauss(self) -> float:
        """Returns the Gauss constant."""
        return (gamma(1 / 4) ** 2) / (2 * sqrt(2 * pi ** 3))
    @property
    def pi_ln2(self) -> float:
        """Returns the value of pi divided by the natural logarithm of 2."""
        return pi / log(2)

    @property
    def pi_log2(self) -> float:
        """Returns the value of pi divided by the base-2 logarithm of 2."""
        return pi / log(2)

    def golden_angle(self, radians: bool = True) -> float:
        """Returns the Golden angle in radians or degrees."""
        return (pi if radians else 180) * (3 - sqrt(5))