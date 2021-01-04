# -*- coding: utf-8 -*-

# lmoments3 library
# Copyright (C) 2012, 2014  J. R. M. Hosking, William Asquith,
# Sam Gillespie, Pierre Gérard-Marchant, Florenz A. P. Hollebrandse
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from collections import OrderedDict
import numpy as np
import scipy.stats
import scipy.stats._continuous_distns
from scipy import special
import math
from .lmoments_base import lmom_ratios_fn


class LmomDistrMixin(object):
    """
    Mixin class to add L-moment methods to :class:`scipy.stats.rv_continous` distribution functions. Distributions using
    the mixin should override the methods :meth:`._lmom_fit` and :meth:`.lmom_ratios`.
    """

    def _lmom_fit(self, lmom_ratios):
        raise NotImplementedError

    def _lmom_ratios(self, *shapes, locs, scale, nmom):
        """
        When overriding, *shapes can be replaced by the actual distribution's shape parameter(s), if any.
        """
        raise NotImplementedError

    def lmom_fit(self, data=[], lmom_ratios=[]):
        """
        Fit the distribution function to the given data or given L-moments.
        :param data: Data to use in calculating the distribution parameters
        :type data: array_like
        :param lmom_ratios: L-moments (ratios) l1, l2, t3, t4, .. to use in calculating the distribution parameters
        :type lmom_ratios: array_like
        :returns: Distribution parameters in `scipy` order, e.g. scale, loc, shape
        :rtype: :class:`OrderedDict`
        """
        n_min = self.numargs + 2
        if len(data) > 0:
            if len(data) <= n_min:
                raise ValueError("At least {} data points must be provided.".format(n_min))
            lmom_ratios = lmom_ratios_fn(data, nmom=n_min)
        elif not lmom_ratios:
            raise Exception("Either `data` or `lmom_ratios` must be provided.")
        elif len(lmom_ratios) < n_min:
            raise ValueError("At least {} number of L-moments must be provided.".format(n_min))

        return self._lmom_fit(lmom_ratios)

    def lmom(self, *args, nmom=5, **kwds):
        """
        Compute the distribution's L-moments, e.g. l1, l2, l3, l4, ..
        :param args: Distribution parameters in order of shape(s), loc, scale
        :type args: float
        :param nmom: Number of moments to calculate
        :type nmom: int
        :param kwds: Distribution parameters as named arguments. See :attr:`rv_continous.shapes` for names of shape
                     parameters
        :type kwds: float
        :returns: List of L-moments
        :rtype: list
        """
        ratios = self.lmom_ratios(*args, nmom=nmom, **kwds)
        moments = ratios[0:2]
        moments += [ratio * moments[1] for ratio in ratios[2:]]
        return moments

    def lmom_ratios(self, *args, nmom=5, **kwds):
        """
        Compute the distribution's L-moment ratios, e.g. l1, l2, t3, t4, ..
        :param args: Distribution parameters in order of shape(s), loc, scale
        :type args: float
        :param nmom: Number of moments to calculate
        :type nmom: int
        :param kwds: Distribution parameters as named arguments. See :attr:`rv_continous.shapes` for names of shape
                     parameters
        :type kwds: float
        :returns: List of L-moment ratios
        :rtype: list
        """
        if nmom > 20:
            return ValueError("Parameter nmom too large. Max of 20.")

        shapes, loc, scale = self._parse_args(*args, **kwds)

        if scale <= 0:
            return ValueError("Invalid scale parameter.")

        return self._lmom_ratios(*shapes, loc=loc, scale=scale, nmom=nmom)

    def nnlf(self, data, *args, **kwds):
        # Override `nnlf` to provide a more consistent interface with shape and loc and scale parameters

        data = np.asarray(data)
        shapes, loc, scale = self._parse_args(*args, **kwds)
        # This is how scipy's nnlf requires parameters
        theta = list(shapes) + [loc, scale]

        # Now call the super class's nnlf
        return scipy.stats.rv_continuous.nnlf(self, x=data, theta=theta)

    def freeze(self, *args, **kwds):
        # Override `freeze` because we're extending the frozen version of the distribution.
        return LmomFrozenDistr(self, *args, **kwds)


class LmomFrozenDistr(scipy.stats.distributions.rv_frozen):
    """
    Frozen version of the distribution returned by :class:`LmomDistrMixin`. Simply provides additional methods supported
    by the mixin.
    """

    def __init__(self, dist, *args, **kwds):
        super().__init__(dist, *args, **kwds)

    def lmom(self, nmom=5):
        return self.dist.lmom(*self.args, nmom=nmom, **self.kwds)

    def lmom_ratios(self, nmom=5):
        return self.dist.lmom_ratios(*self.args, nmom=nmom, **self.kwds)


"""
The following distributions are **not** available in :mod:`scipy.stats`.
"""

class WakebyGen(LmomDistrMixin, scipy.stats.rv_continuous):
    """
    The Wakeby distribution is defined by the transformation:
    (x-xi)/a = (1/b).[1 - (1-U)^b] - (c/d).[1 - (1-U)^(-d)]
    """

    def _argcheck(self, b, c, d):
        b = np.asarray(b)
        c = np.asarray(c)
        d = np.asarray(d)
        check = np.where(b + d > 0,
                         np.where(c == 0, d == 0, True),
                         (b == c) & (c == d) & (d == 0))
        np.putmask(check, c > 0, d > 0)
        np.putmask(check, c < 0, False)
        return check

    def _ppf(self, q, b, c, d):
        z = -np.log(1. - q)
        u = np.where(b == 0, z, (1. - np.exp(-b * z)) / b)
        v = np.where(d == 0, z, (1. - np.exp(d * z)) / d)
        return u - c * v

    def _cdf(self, x, b, c, d):
        if hasattr(x, '__iter__'):
            if hasattr(b, '__iter__'):
                # Assume x, b, c, d are arrays with matching length
                result = np.array([self._cdfwak(_, parameters)
                                   for (_, parameters) in zip(x, zip(b, c, d))])
            else:
                # Only x is an array, paras are scalars
                result = np.array([self._cdfwak(_, [b, c, d])
                                   for _ in x])
        else:
            result = self._cdfwak(x, (b, c, d))
        return result

    def _cdfwak(self, x, para):
        # Only for a single value of x!

        EPS = 1e-8
        MAXIT = 20
        ZINCMX = 3
        ZMULT = 0.2
        UFL = -170
        XI = 0  # stats.rv_continuous deals with scaling
        A = 1  # stats.rv_continuous deals with scaling
        B, C, D = para

        CDFWAK = 0
        if x <= XI:
            return CDFWAK

        # Test for _special cases
        if B == 0 and C == 0 and D == 0:
            Z = (x - XI) / A
            CDFWAK = 1
            if -Z >= UFL:
                CDFWAK = 1 - math.exp(-Z)
            return CDFWAK

        if C == 0:
            CDFWAK = 1
            if x >= (XI + A / B):
                return (CDFWAK)
            Z = -math.log(1 - (x - XI) * B / A) / B
            if -Z >= UFL:
                CDFWAK = 1 - math.exp(-Z)
            return CDFWAK

        if A == 0:
            Z = math.log(1 + (x - XI) * D / C) / D
            if -Z >= UFL:
                CDFWAK = 1 - math.exp(-Z)
            return CDFWAK

        CDFWAK = 1
        if D < 0 and x >= (XI + A / B - C / D):
            return CDFWAK

        Z = 0.7
        if x < self._ppf(0.1, *para):
            Z = 0
        if x < self._ppf(0.99, *para):
            pass
        else:
            if D < 0:
                Z = math.log((x - XI - A / B) * D / C + 1) / D
            if D == 0:
                Z = (x - XI - A / B) / C
            if D > 0:
                Z = math.log((x - XI) * D / C + 1) / D

        for IT in range(1, MAXIT + 1):
            EB = 0
            BZ = -B * Z
            if BZ >= UFL:
                EB = math.exp(BZ)
            GB = Z

            if abs(B) > EPS:
                GB = (1 - EB) / B
            ED = math.exp(D * Z)
            GD = -Z

            if abs(D) > EPS:
                GD = (1 - ED) / D

            XEST = XI + A * GB - C * GD
            FUNC = x - XEST
            DERIV1 = A * EB + C * ED
            DERIV2 = -A * B * EB + C * D * ED
            TEMP = DERIV1 + 0.5 * FUNC * DERIV2 / DERIV1

            if TEMP <= 0:
                TEMP = DERIV1
            ZINC = FUNC / TEMP

            if ZINC > ZINCMX:
                ZINC = ZINCMX

            ZNEW = Z + ZINC

            if ZNEW <= 0:
                Z = Z * ZMULT
            else:
                Z = ZNEW
                if abs(ZINC) <= EPS:
                    CDFWAK = 1
                    if -Z >= UFL:
                        CDFWAK = 1 - math.exp(-Z)
                    return CDFWAK

    def _pdf(self, x, b, c, d):
        t = (1. - self._cdf(x, b, c, d))
        f = t ** (d + 1) / (t ** (b + d) + c)
        return f

    def _lmom_fit(self, lmom_ratios):
        if lmom_ratios[1] <= 0 or abs(lmom_ratios[2]) >= 1 or abs(lmom_ratios[3]) >= 1 or abs(lmom_ratios[4]) >= 1:
            raise ValueError("Invalid L-Moments")

        ALAM1 = lmom_ratios[0]
        ALAM2 = lmom_ratios[1]
        ALAM3 = lmom_ratios[2] * ALAM2
        ALAM4 = lmom_ratios[3] * ALAM2
        ALAM5 = lmom_ratios[4] * ALAM2

        XN1 = 3 * ALAM2 - 25 * ALAM3 + 32 * ALAM4
        XN2 = -3 * ALAM2 + 5 * ALAM3 + 8 * ALAM4
        XN3 = 3 * ALAM2 + 5 * ALAM3 + 2 * ALAM4
        XC1 = 7 * ALAM2 - 85 * ALAM3 + 203 * ALAM4 - 125 * ALAM5
        XC2 = -7 * ALAM2 + 25 * ALAM3 + 7 * ALAM4 - 25 * ALAM5
        XC3 = 7 * ALAM2 + 5 * ALAM3 - 7 * ALAM4 - 5 * ALAM5

        XA = XN2 * XC3 - XC2 * XN3
        XB = XN1 * XC3 - XC1 * XN3
        XC = XN1 * XC2 - XC1 * XN2
        DISC = XB * XB - 4 * XA * XC
        skip20 = False
        if DISC >= 0:
            DISC = math.sqrt(DISC)
            ROOT1 = 0.5 * (-XB + DISC) / XA
            ROOT2 = 0.5 * (-XB - DISC) / XA
            B = max(ROOT1, ROOT2)
            D = -min(ROOT1, ROOT2)
            if D < 1:
                A = (1 + B) * (2 + B) * (3 + B) / (4 * (B + D)) * ((1 + D) * ALAM2 - (3 - D) * ALAM3)
                C = -(1 - D) * (2 - D) * (3 - D) / (4 * (B + D)) * ((1 - B) * ALAM2 - (3 + B) * ALAM3)
                XI = ALAM1 - A / (1 + B) - C / (1 - D)
                skip20 = bool(C >= 0 and A + C >= 0)

        if not skip20:
            D = -(1 - 3 * lmom_ratios[2]) / (1 + lmom_ratios[2])
            C = (1 - D) * (2 - D) * lmom_ratios[1]
            B = 0
            A = 0
            XI = lmom_ratios[0] - C / (1 - D)
            if D <= 0:
                A = C
                B = -D
                C = 0
                D = 0

        para = OrderedDict([('beta', B),
                            ('gamma', C),
                            ('delta', D),
                            ('loc', XI),
                            ('scale', A)])
        return para

    def _lmom_ratios(self, beta, gamma, delta, loc, scale, nmom):
        if (delta >= 1) \
                or (beta + delta <= 0 and (beta != 0 or gamma != 0 or delta != 0)) \
                or (scale == 0 and beta != 0) \
                or (gamma == 0 and delta != 0) \
                or (gamma < 0) \
                or (scale + gamma < 0) \
                or (scale == 0 and gamma == 0):
            raise ValueError("Invalid parameters")

        Y = scale / (1 + beta)
        Z = gamma / (1 - delta)
        xmom = [loc + Y + Z]
        if nmom == 1:
            return xmom

        Y /= (2 + beta)
        Z /= (2 - delta)
        ALAM2 = Y + Z
        xmom.append(ALAM2)
        if nmom == 2:
            return xmom

        for i in range(2, nmom):
            Y *= (i - 1 - beta) / (i + 1 + beta)
            Z *= (i - 1 + delta) / (i + 1 - delta)
            xmom.append((Y + Z) / ALAM2)
        return xmom


wak = WakebyGen(name='wakeby', shapes='beta, gamma, delta')

"""
The following distributions are available in `scipy.stats` and are redefined here with an `LmomDistrMixin` to extend the
scipy distribution with L-moment methods.
"""

class GenParetoGen(LmomDistrMixin, scipy.stats._continuous_distns.genpareto_gen):
    def _lmom_fit(self, lmom_ratios):
        T3 = lmom_ratios[2]
        if lmom_ratios[1] <= 0 or abs(T3) >= 1:
            raise ValueError("L-Moments invalid")

        G = (1 - 3 * T3) / (1 + T3)

        # CHANGE: shape parameter `c` has been negated from original `lmoments` package to be compatible with scipy's
        # GPA distribution function.
        PARA3 = -G
        PARA2 = (1 + G) * (2 + G) * lmom_ratios[1]
        PARA1 = lmom_ratios[0] - PARA2 / (1 + G)

        para = OrderedDict([('c', PARA3),
                            ('loc', PARA1),
                            ('scale', PARA2)])
        return para

    def _lmom_ratios(self, c, loc, scale, nmom):
        # See above, shape parameter negated
        G = -c
        if c > 1:
            raise ValueError("Invalid Parameters")

        Y = 1 / (1 + G)
        xmom = [loc + scale * Y]
        if nmom == 1:
            return xmom

        Y /= (2 + G)
        xmom.append(scale * Y)
        if nmom == 2:
            return xmom

        Y = 1
        for i in range(3, nmom + 1):
            Y *= (i - 2 - G) / (i + G)
            xmom.append(Y)
        return xmom

    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('c', params[0]),
            ('loc', params[1]),
            ('scale', params[2])
        ])
        return param_dict


gpa = GenParetoGen(a=0.0, name='genpareto', shapes='c')


class ExponGen(LmomDistrMixin, scipy.stats._continuous_distns.expon_gen):
    def _lmom_fit(self, lmom_ratios):
        if lmom_ratios[1] <= 0:
            raise ValueError("L-Moments invalid")

        para = OrderedDict([('loc', lmom_ratios[0] - 2 * lmom_ratios[1]),
                            ('scale', 2 * lmom_ratios[1])])
        return para

    def _lmom_ratios(self, loc, scale, nmom):
        xmom = [loc + scale]
        if nmom == 1:
            return xmom

        xmom.append(0.5 * scale)
        if nmom == 2:
            return xmom

        for i in range(3, nmom + 1):
            xmom.append(2 / (i * (i - 1)))

        return xmom

    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('loc', params[0]),
            ('scale', params[1])
        ])
        return param_dict


exp = ExponGen(a=0.0, name='expon')


class GammaGen(LmomDistrMixin, scipy.stats._continuous_distns.gamma_gen):
    def _lmom_fit(self, lmom_ratios):
        A1 = -0.3080
        A2 = -0.05812
        A3 = 0.01765
        B1 = 0.7213
        B2 = -0.5947
        B3 = -2.1817
        B4 = 1.2113

        if lmom_ratios[0] <= lmom_ratios[1] or lmom_ratios[1] <= 0:
            raise ValueError("L-Moments invalid")

        CV = lmom_ratios[1] / lmom_ratios[0]
        if CV >= 0.5:
            T = 1 - CV
            ALPHA = T * (B1 + T * B2) / (1 + T * (B3 + T * B4))
        else:
            T = math.pi * CV ** 2
            ALPHA = (1 + A1 * T) / (T * (1 + T * (A2 + T * A3)))

        para = OrderedDict([('a', ALPHA),
                            ('loc', 0),
                            ('scale', lmom_ratios[0] / ALPHA)])
        return para

    def _lmom_ratios(self, a, loc, scale, nmom):
        A0 = 0.32573501
        [A1, A2, A3] = [0.16869150, 0.078327243, -0.0029120539]
        [B1, B2] = [0.46697102, 0.24255406]
        C0 = 0.12260172
        [C1, C2, C3] = [0.053730130, 0.043384378, 0.011101277]
        [D1, D2] = [0.18324466, 0.20166036]
        [E1, E2, E3] = [2.3807576, 1.5931792, 0.11618371]
        [F1, F2, F3] = [5.1533299, 7.1425260, 1.9745056]

        if a <= 0:
            raise ValueError("Invalid Parameters")
        if nmom > 4:
            raise ValueError("Parameter nmom too large")
        if loc != 0:
            raise ValueError("Location parameter not supported for Gamma distribution.")

        xmom = []
        xmom.append(a * scale)
        if nmom == 1:
            return (xmom)

        xmom.append(scale * 1 / math.sqrt(math.pi) * math.exp(special.gammaln(a + 0.5) - special.gammaln(a)))
        if nmom == 2:
            return (xmom)

        if a < 1:
            Z = a
            xmom.append((((E3 * Z + E2) * Z + E1) * Z + 1) / (((F3 * Z + F2) * Z + F1) * Z + 1))
            if nmom == 3:
                return (xmom)
            xmom.append((((C3 * Z + C2) * Z + C1) * Z + C0) / ((D2 * Z + D1) * Z + 1))
            if nmom == 4:
                return (xmom)
        else:
            Z = 1 / a
            xmom.append(math.sqrt(Z) * (((A3 * Z + A2) * Z + A1) * Z + A0) / ((B2 * Z + B1) * Z + 1))
            if nmom == 3:
                return (xmom)

            xmom.append((((C3 * Z + C2) * Z + C1) * Z + C0) / ((D2 * Z + D1) * Z + 1))
            if nmom == 4:
                return (xmom)

    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('a', params[0]),
            ('loc', params[1]),
            ('scale', params[2])
        ])
        return param_dict


gam = GammaGen(a=0.0, name='gamma', shapes='a')


class GenextremeGen(LmomDistrMixin, scipy.stats._continuous_distns.genextreme_gen):
    def _lmom_fit(self, lmom_ratios):
        SMALL = 1e-5
        eps = 1e-6
        maxit = 20
        EU = 0.57721566
        DL2 = math.log(2)
        DL3 = math.log(3)
        A0 = 0.28377530
        A1 = -1.21096399
        A2 = -2.50728214
        A3 = -1.13455566
        A4 = -0.07138022
        B1 = 2.06189696
        B2 = 1.31912239
        B3 = 0.25077104
        C1 = 1.59921491
        C2 = -0.48832213
        C3 = 0.01573152
        D1 = -0.64363929
        D2 = 0.08985247

        T3 = lmom_ratios[2]
        if lmom_ratios[1] <= 0 or abs(T3) >= 1:
            raise ValueError("L-Moments Invalid")

        if T3 <= 0:
            G = (A0 + T3 * (A1 + T3 * (A2 + T3 * (A3 + T3 * A4)))) / (1 + T3 * (B1 + T3 * (B2 + T3 * B3)))

            if T3 >= -0.8:
                para3 = G
                GAM = math.exp(special.gammaln(1 + G))
                para2 = lmom_ratios[1] * G / (GAM * (1 - 2 ** -G))
                para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
                para = OrderedDict([('c', para3),
                                    ('loc', para1),
                                    ('scale', para2)])
                return para
            elif T3 <= -0.97:
                G = 1 - math.log(1 + T3) / DL2

            T0 = (T3 + 3) * 0.5
            for IT in range(1, maxit):
                X2 = 2 ** -G
                X3 = 3 ** -G
                XX2 = 1 - X2
                XX3 = 1 - X3
                T = XX3 / XX2
                DERIV = (XX2 * X3 * DL3 - XX3 * X2 * DL2) / (XX2 ** 2)
                GOLD = G
                G -= (T - T0) / DERIV

                if abs(G - GOLD) <= eps * G:
                    para3 = G
                    GAM = math.exp(special.gammaln(1 + G))
                    para2 = lmom_ratios[1] * G / (GAM * (1 - 2 ** -G))
                    para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
                    para = OrderedDict([('c', para3),
                                        ('loc', para1),
                                        ('scale', para2)])
                    return para
            raise Exception("Iteration has not converged")
        else:
            Z = 1 - T3
            G = (-1 + Z * (C1 + Z * (C2 + Z * C3))) / (1 + Z * (D1 + Z * D2))
            if abs(G) < SMALL:
                para2 = lmom_ratios[1] / DL2
                para1 = lmom_ratios[0] - EU * para2
                para = OrderedDict([('c', 0),
                                    ('loc', para1),
                                    ('scale', para2)])
            else:
                para3 = G
                GAM = math.exp(special.gammaln(1 + G))
                para2 = lmom_ratios[1] * G / (GAM * (1 - 2 ** -G))
                para1 = lmom_ratios[0] - para2 * (1 - GAM) / G
                para = OrderedDict([('c', para3),
                                    ('loc', para1),
                                    ('scale', para2)])
            return para

    def _lmom_ratios(self, c, loc, scale, nmom):
        ZMOM = [0.577215664901532861, 0.693147180559945309,
                0.169925001442312363, 0.150374992788438185,
                0.558683500577583138e-1, 0.581100239999710876e-1,
                0.276242584297309125e-1, 0.305563766579053126e-1,
                0.164650282258328802e-1, 0.187846624298170912e-1,
                0.109328215063027148e-1, 0.126973126676329530e-1,
                0.778982818057231804e-2, 0.914836179621999726e-2,
                0.583332389328363588e-2, 0.690104287590348154e-2,
                0.453267970180679549e-2, 0.538916811326595459e-2,
                0.362407767772390e-2, 0.432387608605538096e-2]
        SMALL = 1e-6
        if c <= -1:
            raise ValueError("Invalid Parameters")

        if abs(c) > SMALL:
            GAM = math.exp(special.gammaln(1 + c))
            xmom = [loc + scale * (1 - GAM) / c]
            if nmom == 1:
                return xmom

            XX2 = 1 - 2 ** (-c)
            xmom.append(scale * XX2 * GAM / c)
            if nmom == 2:
                return xmom

            Z0 = 1
            for j in range(2, nmom):
                DJ = j + 1
                BETA = (1 - DJ ** -c) / XX2
                Z0 *= (4 * DJ - 6) / DJ
                Z = Z0 * 3 * (DJ - 1) / (DJ + 1)
                SUM = Z0 * BETA - Z
                if j == 2:
                    xmom.append(SUM)
                else:
                    for i in range(1, j - 1):
                        DI = i + 1
                        Z *= (DI + DI + 1) * (DJ - DI) / ((DI + DI - 1) * (DJ + DI))
                        SUM -= Z * xmom[i + 1]
                    xmom.append(SUM)
            return xmom

        else:
            xmom = [loc]
            if nmom == 1:
                return xmom

            xmom.append(scale * ZMOM[1])
            if nmom == 2:
                return xmom

            for i in range(2, nmom):
                xmom.append(ZMOM[i - 1])

            return xmom

    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('c', params[0]),
            ('loc', params[1]),
            ('scale', params[2])
        ])
        return param_dict


gev = GenextremeGen(name='genextreme', shapes='c')


class GumbelGen(LmomDistrMixin, scipy.stats._continuous_distns.gumbel_r_gen):
    def _lmom_fit(self, lmom_ratios):
        EU = 0.577215664901532861
        if lmom_ratios[1] <= 0:
            raise ValueError("L-Moments Invalid")

        para2 = lmom_ratios[1] / math.log(2)
        para1 = lmom_ratios[0] - EU * para2
        para = OrderedDict([('loc', para1),
                            ('scale', para2)])
        return para

    def _lmom_ratios(self, loc, scale, nmom):
        ZMOM = [0.577215664901532861, 0.693147180559945309, 0.169925001442312363, 0.150374992788438185,
                0.0558683500577583138, 0.0581100239999710876, 0.0276242584297309125, 0.0305563766579053126,
                0.0164650282258328802, 0.0187846624298170912, 0.0109328215063027148, 0.0126973126676329530,
                0.00778982818057231804, 0.00914836179621999726, 0.00583332389328363588, 0.00690104287590348154,
                0.00453267970180679549, 0.00538916811326595459, 0.00362407767772368790, 0.00432387608605538096]
        xmom = [loc + scale * ZMOM[0]]
        if nmom == 1:
            return xmom

        xmom.append(scale * ZMOM[1])
        if nmom == 2:
            return xmom

        for i in range(2, nmom):
            xmom.append(ZMOM[i])
        return xmom

    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('loc', params[0]),
            ('scale', params[1])
        ])
        return param_dict


gum = GumbelGen(name='gumbel_r')


class NormGen(LmomDistrMixin, scipy.stats._continuous_distns.norm_gen):
    def _lmom_fit(self, lmom_ratios):
        if lmom_ratios[1] <= 0:
            raise ValueError("L-Moments invalid")

        para = OrderedDict([('loc', lmom_ratios[0]),
                            ('scale', lmom_ratios[1] * math.sqrt(math.pi))])
        return para

    def _lmom_ratios(self, *shapes, loc, scale, nmom):
        ZMOM = [0, 0.564189583547756287, 0, 0.122601719540890947, 0, 0.0436611538950024944, 0, 0.0218431360332508776, 0,
                0.0129635015801507746, 0, 0.00852962124191705402, 0, 0.00601389015179323333, 0, 0.00445558258647650150,
                0, 0.00342643243578076985, 0, 0.00271267963048139365]
        xmom = [loc]
        if nmom == 1:
            return xmom

        xmom.append(scale * ZMOM[1])
        if nmom == 2:
            return xmom

        for M in range(2, nmom):
            xmom.append(ZMOM[M])

        return xmom

    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('loc', params[0]),
            ('scale', params[1])
        ])
        return param_dict


nor = NormGen(name='norm')


class Pearson3Gen(LmomDistrMixin, scipy.stats._continuous_distns.pearson3_gen):
    def _lmom_fit(self, lmom_ratios):
        Small = 1e-6
        # Constants used in Minimax Approx:
        C1 = 0.2906
        C2 = 0.1882
        C3 = 0.0442
        D1 = 0.36067
        D2 = -0.59567
        D3 = 0.25361
        D4 = -2.78861
        D5 = 2.56096
        D6 = -0.77045

        T3 = abs(lmom_ratios[2])
        if lmom_ratios[1] <= 0 or T3 >= 1:
            raise ValueError("L-Moments invalid")

        if T3 <= Small:
            loc = lmom_ratios[0]
            scale = lmom_ratios[1] * math.sqrt(math.pi)
            skew = 0
        else:
            if T3 >= (1.0 / 3):
                T = 1 - T3
                Alpha = T * (D1 + T * (D2 + T * D3)) / (1 + T * (D4 + T * (D5 + T * D6)))
            else:
                T = 3 * math.pi * T3 * T3
                Alpha = (1 + C1 * T) / (T * (1 + T * (C2 + T * C3)))

            RTALPH = math.sqrt(Alpha)
            BETA = math.sqrt(math.pi) * lmom_ratios[1] * math.exp(special.gammaln(Alpha) - special.gammaln(Alpha + 0.5))
            loc = lmom_ratios[0]
            scale = BETA * RTALPH
            skew = 2 / RTALPH
            if lmom_ratios[2] < 0:
                skew *= -1

        return OrderedDict([('skew', skew),
                            ('loc', loc),
                            ('scale', scale)])

    def _lmom_ratios(self, skew, loc, scale, nmom):
        SMALL = 1e-6
        CONST = 1 / math.sqrt(math.pi)
        A0 = 0.32573501
        [A1, A2, A3] = [0.16869150, 0.078327243, -0.0029120539]
        [B1, B2] = [0.46697102, 0.24255406]
        C0 = 0.12260172
        [C1, C2, C3] = 0.053730130, 0.043384378, 0.011101277
        [D1, D2] = [0.18324466, 0.20166036]
        [E1, E2, E3] = [2.3807576, 1.5931792, 0.11618371]
        [F1, F2, F3] = [5.1533299, 7.1425260, 1.9745056]
        [G1, G2, G3] = [2.1235833, 4.1670213, 3.1925299]
        [H1, H2, H3] = [9.0551443, 26.649995, 26.193668]

        # Calculate only up to 4 L-moments for Pearson 3
        if nmom > 4:
            raise ValueError("Parameter nmom too large")

        xmom = [loc]
        if nmom == 1:
            return xmom

        if abs(skew) < SMALL:
            xmom = [loc]
            if nmom == 1:
                return xmom

            xmom.append(CONST * scale)
            if nmom == 2:
                return xmom

            xmom.append(0)
            if nmom == 3:
                return xmom

            xmom.append(C0)
            return xmom
        else:
            Alpha = 4 / (skew * skew)
            Beta = abs(0.5 * scale * skew)
            ALAM2 = CONST * math.exp(special.gammaln(Alpha + 0.5) - special.gammaln(Alpha))
            xmom.append(ALAM2 * Beta)
            if nmom == 2:
                return xmom

            if Alpha < 1:
                Z = Alpha
                xmom.append((((E3 * Z + E2) * Z + E1) * Z + 1) / (((F3 * Z + F2) * Z + F1) * Z + 1))
                if skew < 0:
                    xmom[2] *= -1
                if nmom == 3:
                    return xmom

                xmom.append((((G3 * Z + G2) * Z + G1) * Z + 1) / (((H3 * Z + H2) * Z + H1) * Z + 1))
                return xmom
            else:
                Z = 1.0 / Alpha
                xmom.append(math.sqrt(Z) * (((A3 * Z + A2) * Z + A1) * Z + A0) / ((B2 * Z + B1) * Z + 1))
                if skew < 0:
                    xmom[2] *= -1
                if nmom == 3:
                    return xmom

                xmom.append((((C3 * Z + C2) * Z + C1) * Z + C0) / ((D2 * Z + D1) * Z + 1))
                return xmom

    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('skew', params[0]),
            ('loc', params[1]),
            ('scale', params[2])
        ])
        return param_dict


pe3 = Pearson3Gen(name="pearson3", shapes='skew')


class WeibullMinGen(LmomDistrMixin, scipy.stats._continuous_distns.weibull_min_gen):
    def _lmom_fit(self, lmom_ratios):
        if lmom_ratios[1] <= 0 or lmom_ratios[2] >= 1 or lmom_ratios[2] <= -gum.lmom_ratios(nmom=3)[2]:
            raise ValueError("L-Moments invalid")

        pg = gev.lmom_fit(lmom_ratios=[-lmom_ratios[0], lmom_ratios[1], -lmom_ratios[2]])
        delta = 1 / pg['c']
        beta = pg['scale'] / pg['c']
        para = OrderedDict([('c', delta),
                            ('loc', -pg['loc'] - beta),
                            ('scale', beta)])
        return para

    def _lmom_ratios(self, c, loc, scale, nmom):
        if c <= 0:
            raise ValueError("Invalid parameters")

        xmom = gev.lmom_ratios(loc=0, scale=scale / c, c=1 / c, nmom=nmom)
        xmom[0] = loc + scale - xmom[0]
        xmom[2] *= -1
        return xmom

    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('c', params[0]),
            ('loc', params[1]),
            ('scale', params[2])
        ])
        return param_dict


wei = WeibullMinGen(a=0.0, name='weibull_min', shapes='c')


class GenLogistic(LmomDistrMixin, scipy.stats._continuous_distns.genlogistic_gen):
    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('c', params[0]),
            ('loc', params[1]),
            ('scale', params[2])
        ])
        return param_dict


glo = GenLogistic(name='genlogistic')


class GenNormal(LmomDistrMixin, scipy.stats._continuous_distns.gennorm_gen):
    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('beta', params[0]),
            ('loc', params[1]),
            ('scale', params[2])
        ])
        return param_dict


gno = GenNormal(name='gennorm')


class Kappa4(LmomDistrMixin, scipy.stats._continuous_distns.kappa4_gen):
    def fit(self, data, *args, **kwargs):
        params = super().fit(data, *args, **kwargs)

        param_dict = OrderedDict([
            ('h', params[0]),
            ('k', params[1]),
            ('loc', params[2]),
            ('scale', params[3])
        ])
        return param_dict


kap = Kappa4(name='kappa4')

