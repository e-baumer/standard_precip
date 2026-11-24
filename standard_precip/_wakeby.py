"""Wakeby distribution in Hosking's parameterization.

The Wakeby distribution has no closed-form CDF. It is defined through its
quantile function (Hosking and Wallis, 1997, Regional Frequency Analysis,
section A.11)

    x(F) = xi + alpha/beta * (1 - (1 - F)**beta)
              - gamma/delta * (1 - (1 - F)**(-delta))

with the parameters named here as lmoments3 names them: loc (xi), scale
(alpha), beta, gamma and delta. Valid parameters satisfy either
beta + delta > 0 or beta = gamma = delta = 0, together with delta < 1,
gamma >= 0, alpha + gamma >= 0, alpha = 0 implying beta = 0, and gamma = 0
implying delta = 0. A negative delta is valid and gives a distribution that
is bounded above.

Parameters are estimated by lmoments3 (Hosking's PELWAK), which is correct.
The CDF, PDF and quantile function are evaluated here and not by
lmoments3.distr.wak, because that scipy-based object is wrong for fitted
parameters (verified against lmoments3 1.0.8 on 2026-11-24, reported upstream
as Ouranosinc/lmoments3#34). It multiplies gamma by alpha a second time, since
scipy applies ``scale`` to the whole standardized quantile while gamma is
already expressed in data units, and its argument check rejects every fit
with delta < 0 and every fit with alpha = 0, returning NaN. On
data/monthly_data.csv that left 9 of 12 calendar months entirely NaN and the
remaining months off by up to about 2 index units.

The CDF is obtained by inverting the quantile function with bisection on
z = -log(1 - F), the variable Hosking's CDFWAK also solves for. The quantile
function is nondecreasing in z for every valid parameter set, so bisection
always converges, and working in z keeps full relative precision in the upper
tail. z is searched on [0, 40]; F = 1 - exp(-40) differs from 1 by less than
float64 resolution, so nothing representable is lost.
"""

import numpy as np
from lmoments3 import distr as lm3_distr

_Z_MAX = 40.0
_BISECTIONS = 64


class Wakeby:
    """Wakeby distribution exposing the subset of the scipy distribution
    interface that the registry in _distributions relies on: lmom_fit, cdf,
    pdf, ppf and numargs. All evaluation methods take the named parameters
    returned by lmom_fit."""

    numargs = 3

    def lmom_fit(self, data, **kwargs):
        """Fit by L-moments and return an OrderedDict with keys beta, gamma,
        delta, loc and scale."""
        return lm3_distr.wak.lmom_fit(data, **kwargs)

    @staticmethod
    def _valid(beta, gamma, delta, scale):
        """Hosking's parameter constraints, under which the quantile function
        is nondecreasing."""
        all_zero = beta == 0 and gamma == 0 and delta == 0
        return bool(
            np.all(np.isfinite([beta, gamma, delta, scale]))
            and (beta + delta > 0 or all_zero)
            and delta < 1
            and gamma >= 0
            and scale + gamma >= 0
            and not (scale == 0 and beta != 0)
            and not (gamma == 0 and delta != 0)
        )

    @staticmethod
    def _quantile(z, beta, gamma, delta, loc, scale):
        """Quantile function in terms of z = -log(1 - F), written as in
        Hosking's QUAWAK: xi + alpha * (1 - exp(-beta * z)) / beta
        + gamma * (exp(delta * z) - 1) / delta, where each fraction tends to
        z as its exponent tends to zero. A term whose coefficient is zero is
        dropped so that it cannot produce 0 * inf at z = inf."""
        z = np.asarray(z, dtype=float)
        x = np.full(z.shape, float(loc))
        with np.errstate(over="ignore"):
            if scale != 0:
                x = x + scale * (z if beta == 0 else -np.expm1(-beta * z) / beta)
            if gamma != 0:
                x = x + gamma * (z if delta == 0 else np.expm1(delta * z) / delta)
        return x

    def _solve_z(self, x, beta, gamma, delta, loc, scale):
        """Return z = -log(1 - F(x)) for each x, with z = 0 at or below the
        lower bound, z = inf at or above the largest representable quantile,
        and NaN where x is NaN or the parameters are invalid."""
        x = np.asarray(x, dtype=float)
        if not self._valid(beta, gamma, delta, scale) or not np.isfinite(loc):
            return np.full(x.shape, np.nan)

        params = (beta, gamma, delta, loc, scale)
        lo = np.zeros(x.shape)
        hi = np.full(x.shape, _Z_MAX)
        for _ in range(_BISECTIONS):
            mid = 0.5 * (lo + hi)
            below = self._quantile(mid, *params) < x
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
        z = 0.5 * (lo + hi)

        z = np.where(x <= loc, 0.0, z)
        z = np.where(x >= self._quantile(_Z_MAX, *params), np.inf, z)
        return np.where(np.isnan(x), np.nan, z)

    def cdf(self, x, beta, gamma, delta, loc=0.0, scale=1.0):
        """Cumulative distribution function: 0 at or below the lower bound
        loc, 1 at or above the upper bound when one exists."""
        return -np.expm1(-self._solve_z(x, beta, gamma, delta, loc, scale))

    def pdf(self, x, beta, gamma, delta, loc=0.0, scale=1.0):
        """Probability density, the reciprocal of the derivative of the
        quantile function: (1 - F) / (alpha * (1 - F)**beta
        + gamma * (1 - F)**(-delta)). Zero outside the support."""
        z = self._solve_z(x, beta, gamma, delta, loc, scale)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            density = np.exp(-z) / (scale * np.exp(-beta * z) + gamma * np.exp(delta * z))
        outside = (np.asarray(x, dtype=float) < loc) | np.isinf(z)
        return np.where(outside, 0.0, density)

    def ppf(self, q, beta, gamma, delta, loc=0.0, scale=1.0):
        """Quantile function x(F). NaN for probabilities outside [0, 1] and
        for invalid parameters."""
        q = np.asarray(q, dtype=float)
        if not self._valid(beta, gamma, delta, scale) or not np.isfinite(loc):
            return np.full(q.shape, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = -np.log1p(-q)
            x = self._quantile(z, beta, gamma, delta, loc, scale)
        return np.where((q >= 0) & (q <= 1), x, np.nan)


wak = Wakeby()
