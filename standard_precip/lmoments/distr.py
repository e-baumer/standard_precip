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


class GenlogisticGen(LmomDistrMixin, scipy.stats.rv_continuous):
    """
    The CDF is given by
    .. math::
       F(x;k) = \\frac{1}{1 + \left[1 - kx\\right]^{1/k}}
    """

    def _argcheck(self, k):
        return (k == k)

    def _cdf(self, x, k):
        u = np.where(k == 0, np.exp(-x), (1. - k * x) ** (1. / k))
        return 1. / (1. + u)

    def _pdf(self, x, k):
        u = np.where(k == 0, np.exp(-x), (1. - k * x) ** (1. / k))
        return u ** (1. - k) / (1. + u) ** 2

    def _ppf(self, q, k):
        F = q / (1. - q)
        return np.where(k == 0, np.log(F), (1 - F ** (-k)) / k)

    def _lmom_fit(self, lmom_ratios):
        SMALL = 1e-6
        G = -lmom_ratios[2]
        if lmom_ratios[1] <= 0 or abs(G) >= 1:
            raise ValueError("L-Moments invalid")

        if abs(G) <= SMALL:
            G = 0
            para1 = lmom_ratios[0]
            A = lmom_ratios[1]
        else:
            GG = G * math.pi / math.sin(G * math.pi)
            A = lmom_ratios[1] / GG
            para1 = lmom_ratios[0] - A * (1 - GG) / G

        para = OrderedDict([('k', G),
                            ('loc', para1),
                            ('scale', A)])
        return para

    def _lmom_ratios(self, k, loc, scale, nmom):
        if abs(k) >= 1:
            return ValueError("Invalid parameters")

        SMALL = 1e-4
        C1 = math.pi ** 2 / 6
        C2 = 7 * math.pi ** 4 / 360

        Z = [[0], [0], [1]]
        Z.append([0.166666666666666667, 0.833333333333333333])
        Z.append([0.416666666666666667, 0.583333333333333333])
        Z.append([0.666666666666666667e-1, 0.583333333333333333, 0.350000000000000000])
        Z.append([0.233333333333333333, 0.583333333333333333, 0.183333333333333333])
        Z.append([0.357142857142857143e-1, 0.420833333333333333, 0.458333333333333333, 0.851190476190476190e-1])
        Z.append([0.150992063492063492, 0.515625000000000000, 0.297916666666666667, 0.354662698412698413e-1])
        Z.append([0.222222222222222222e-1, 0.318893298059964727, 0.479976851851851852, 0.165509259259259259,
                  0.133983686067019400e-1])
        Z.append([0.106507936507936508, 0.447663139329805996, 0.360810185185185185, 0.803902116402116402e-1,
                  0.462852733686067019e-2])
        Z.append([0.151515151515151515e-1, 0.251316137566137566, 0.469695216049382716, 0.227650462962962963,
                  0.347139550264550265e-1, 0.147271324354657688e-2])
        Z.append([0.795695045695045695e-1, 0.389765946502057613, 0.392917309670781893, 0.123813106261022928,
                  0.134998713991769547e-1, 0.434261597456041900e-3])
        Z.append([0.109890109890109890e-1, 0.204132996632996633, 0.447736625514403292, 0.273053442827748383,
                  0.591917438271604938e-1, 0.477687757201646091e-2, 0.119302636663747775e-3])
        Z.append([0.619345205059490774e-1, 0.342031759392870504, 0.407013705173427396, 0.162189192806752331,
                  0.252492100235155791e-1, 0.155093427662872107e-2, 0.306778208563922850e-4])
        Z.append([0.833333333333333333e-2, 0.169768364902293474, 0.422191282868366202, 0.305427172894620811,
                  0.840827939972285210e-1, 0.972435791446208113e-2, 0.465280282988616322e-3, 0.741380670696146887e-5])
        Z.append([0.497166028416028416e-1, 0.302765838589871328, 0.410473300089185506, 0.194839026503251764,
                  0.386598063704648526e-1, 0.341399407642897226e-2, 0.129741617371825705e-3, 0.168991182291033482e-5])
        Z.append([0.653594771241830065e-2, 0.143874847595085690, 0.396432853710259464, 0.328084180720899471,
                  0.107971393165194318, 0.159653369932077769e-1, 0.110127737569143819e-2, 0.337982364582066963e-4,
                  0.364490785333601627e-6])
        Z.append([0.408784570549276431e-1, 0.270244290725441519, 0.407599524514551521, 0.222111426489320008,
                  0.528463884629533398e-1, 0.598298239272872761e-2, 0.328593965565898436e-3, 0.826179113422830354e-5,
                  0.746033771150646605e-7])
        Z.append([0.526315789473684211e-2, 0.123817655753054913, 0.371859291444794917, 0.343568747670189607,
                  0.130198662812524058, 0.231474364899477023e-1, 0.205192519479869981e-2, 0.912058258107571930e-4,
                  0.190238611643414884e-5, 0.145280260697757497e-7])

        GG = k * k
        if abs(k) > SMALL:
            ALAM2 = k * math.pi / math.sin(k * math.pi)
            ALAM1 = (1 - ALAM2) / k
        else:
            ALAM1 = -k * (C1 + GG * C2)
            ALAM2 = 1 + GG * (C1 + GG * C2)

        xmom = [loc + scale * ALAM1]
        if nmom == 1:
            return xmom

        xmom.append(scale * ALAM2)
        if nmom == 2:
            return xmom

        for M in range(3, nmom + 1):
            kmax = M // 2
            SUMM = Z[M - 1][kmax - 1]
            for K in range(kmax - 1, 0, -1):
                SUMM = SUMM * GG + Z[M - 1][K - 1]
            if M % 2 > 0:
                SUMM *= -k
            xmom.append(SUMM)

        return xmom


glo = GenlogisticGen(name='glogistic', shapes='k')


class GennormGen(LmomDistrMixin, scipy.stats.rv_continuous):
    """
    The CDF is given by
    .. math::
       F(x) = \Phi{\left[ -k^{-1} \log\{1 - kx\} \\right]}
    """

    def _argcheck(self, k):
        return (k == k)

    def _cdf(self, x, k):
        y = np.where(k == 0, x, -np.log(1. - k * x) / k)
        return 0.5 * (1 + special.erf(y * np.sqrt(0.5)))

    def _pdf(self, x, k):
        u = np.where(k == 0, x, -np.log(1. - k * x) / k)
        return np.exp(k * u - u * u / 2.) / np.sqrt(2 * np.pi)

    def _ppf(self, q, k):
        u = special.ndtri(q)  # Normal distribution's ppf
        return np.where(k == 0, u, (1. - np.exp(-k * u)) / k)

    def _lmom_fit(self, lmom_ratios):
        SMALL = 1e-8
        A0 = 0.20466534e+01
        A1 = -0.36544371e+01
        A2 = 0.18396733e+01
        A3 = -0.20360244e+00
        B1 = -0.20182173e+01
        B2 = 0.12420401e+01
        B3 = -0.21741801e+00

        T3 = lmom_ratios[2]
        if lmom_ratios[1] <= 0 or abs(T3) >= 1:
            raise ValueError("L-Moments invalid")

        if abs(T3) >= 0.95:
            U, A, G = 0, -1, 0
        elif abs(T3) <= SMALL:
            U, A, G = lmom_ratios[0], lmom_ratios[1] * math.sqrt(math.pi), 0
        else:
            TT = T3 ** 2
            G = -T3 \
                * (A0 + TT * (A1 + TT * (A2 + TT * A3))) \
                / (1 + TT * (B1 + TT * (B2 + TT * B3)))
            E = math.exp(0.5 * G ** 2)
            A = lmom_ratios[1] * G / (E * special.erf(0.5 * G))
            U = lmom_ratios[0] + A * (E - 1) / G
        para = OrderedDict([('k', G),
                            ('loc', U),
                            ('scale', A)])
        return para

    def _lmom_ratios(self, k, loc, scale, nmom):
        ZMOM = [0, 0.564189583547756287, 0, 0.122601719540890947,
                0, 0.436611538950024944e-1, 0, 0.218431360332508776e-1,
                0, 0.129635015801507746e-1, 0, 0.852962124191705402e-2,
                0, 0.601389015179323333e-2, 0, 0.445558258647650150e-2,
                0, 0.342643243578076985e-2, 0, 0.271267963048139365e-2]

        RRT2 = 1 / math.sqrt(2)
        RRTPI = 1 / math.sqrt(math.pi)
        RANGE = 5
        EPS = 1e-8
        MAXIT = 10

        if abs(k) <= EPS:
            xmom = [loc]
            if nmom == 1:
                return xmom

            xmom.append(scale * ZMOM[1])
            if nmom == 2:
                return xmom

            for i in range(3, nmom + 1):
                xmom.append(ZMOM[i - 1])
            return xmom
        else:
            EGG = math.exp(0.5 * k ** 2)
            ALAM1 = (1 - EGG) / k
            xmom = [loc + scale * ALAM1]
            if nmom == 1:
                return xmom

            ALAM2 = EGG * special.erf(0.5 * k) / k
            xmom.append(scale * ALAM2)
            if nmom == 2:
                return xmom

            CC = -k * RRT2
            XMIN = CC - RANGE
            XMAX = CC + RANGE
            SUMM = [0] * nmom

            N = 16
            XINC = (XMAX - XMIN) / N

            for i in range(1, N):
                X = XMIN + i * XINC
                E = math.exp(-((X - CC) ** 2))
                D = special.erf(X)
                P1 = 1
                P = D
                for m in range(3, nmom + 1):
                    C1 = m + m - 3
                    C2 = m - 2
                    C3 = m - 1
                    P2 = P1
                    P1 = P
                    P = (C1 * D * P1 - C2 * P2) / C3
                    SUMM[m - 1] += E * P

            EST = []
            for i in SUMM:
                EST.append(i * XINC)

            for _ in range(MAXIT):
                ESTX = EST
                N *= 2
                XINC = (XMAX - XMIN) / N
                for i in range(1, N - 1, 2):
                    X = XMIN + i * XINC
                    E = math.exp(-((X - CC) ** 2))
                    D = special.erf(X)
                    P1 = 1
                    P = D
                    for m in range(3, nmom + 1):
                        C1 = m + m - 3
                        C2 = m - 2
                        C3 = m - 1
                        P2 = P1
                        P1 = P
                        P = (C1 * D * P1 - C2 * P2) / C3
                        SUMM[m - 1] += E * P

                NOTCGD = 0
                for m in range(nmom, 2, -1):
                    EST[m - 1] = SUMM[m - 1] * XINC
                    if abs(EST[m - 1] - ESTX[m - 1]) > EPS * abs(EST[m - 1]):
                        NOTCGD = m

                if NOTCGD == 0:
                    CONST = -math.exp(CC ** 2) * RRTPI / (ALAM2 * k)

                    for m in range(3, nmom + 1):
                        xmom.append(CONST * EST[m - 1])
                    return xmom
                else:
                    raise Exception("L-moment ratios computation did not converge.")


gno = GennormGen(name='gennorm', shapes='k')


class KappaGen(LmomDistrMixin, scipy.stats.rv_continuous):
    """
    The CDF is given by
    .. math::
       F(x; a, b) = \left[1-h\{1-kx\}^{1/k}\\right]^{1/h}
    """

    def _argcheck(self, k, h):
        k = np.asarray(k)
        h = np.asarray(h)
        # Upper bound
        self.b = np.where(k <= 0, np.inf, 1. / k)
        # Lower bound
        self.a = np.where(h > 0,
                          np.where(k == 0, 0., (1 - h ** (-k)) / k),
                          np.where(k < 0, 1. / k, -np.inf))
        return (k == k) | (h == h)

    def _cdf(self, x, k, h):
        y = np.where(k == 0, np.exp(-x), (1 - k * x) ** (1. / k))
        return np.where(h == 0, np.exp(-y), (1. - h * y) ** (1. / h))

    def _pdf(self, x, k, h):
        y = (1 - k * x) ** (1. / k - 1.)
        y *= self._cdf(x, k, h) ** (1. - h)
        return y

    def _ppf(self, q, k, h):
        y = np.where(h == 0, -np.log(q), (1. - q ** h) / h)
        y = np.where(k == 0, -np.log(y), (1. - y ** k) / k)
        return y

    def _lmom_fit(self, lmom_ratios):
        EPS = 1e-6
        MAXIT = 20
        MAXSR = 10
        HSTART = 1.001
        BIG = 10
        OFLEXP = 170
        OFLGAM = 53

        T3 = lmom_ratios[2]
        T4 = lmom_ratios[3]
        if lmom_ratios[1] <= 0 or abs(T3) >= 1 or abs(T4) >= 1 or T4 <= (5 * T3 * T3 - 1) / 4 or \
                        T4 >= (5 * T3 * T3 + 1) / 6:
            raise ValueError("L-Moments invalid")

        G = (1 - 3 * T3) / (1 + T3)
        H = HSTART
        Z = G + H * 0.725
        Xdist = BIG

        # Newton-Raphson Iteration
        for it in range(1, MAXIT + 1):
            for i in range(1, MAXSR + 1):
                if G > OFLGAM:
                    raise Exception("Failed to converge")
                if H > 0:
                    U1 = math.exp(special.gammaln(1 / H) - special.gammaln(1 / H + 1 + G))
                    U2 = math.exp(special.gammaln(2 / H) - special.gammaln(2 / H + 1 + G))
                    U3 = math.exp(special.gammaln(3 / H) - special.gammaln(3 / H + 1 + G))
                    U4 = math.exp(special.gammaln(4 / H) - special.gammaln(4 / H + 1 + G))
                else:
                    U1 = math.exp(special.gammaln(-1 / H - G) - special.gammaln(-1 / H + 1))
                    U2 = math.exp(special.gammaln(-2 / H - G) - special.gammaln(-2 / H + 1))
                    U3 = math.exp(special.gammaln(-3 / H - G) - special.gammaln(-3 / H + 1))
                    U4 = math.exp(special.gammaln(-4 / H - G) - special.gammaln(-4 / H + 1))

                ALAM2 = U1 - 2 * U2
                ALAM3 = -U1 + 6 * U2 - 6 * U3
                ALAM4 = U1 - 12 * U2 + 30 * U3 - 20 * U4
                if ALAM2 == 0:
                    raise Exception("Failed to converge")
                TAU3 = ALAM3 / ALAM2
                TAU4 = ALAM4 / ALAM2
                E1 = TAU3 - T3
                E2 = TAU4 - T4

                DIST = max(abs(E1), abs(E2))
                if DIST < Xdist:
                    Success = 1
                    break
                else:
                    DEL1 = 0.5 * DEL1
                    DEL2 = 0.5 * DEL2
                    G = XG - DEL1
                    H = XH - DEL2

            if Success == 0:
                raise Exception("Failed to converge")

            # Test for convergence
            if DIST < EPS:
                TEMP = special.gammaln(1 + G)
                if TEMP > OFLEXP:
                    raise Exception("Failed to converge")
                GAM = math.exp(TEMP)
                TEMP = (1 + G) * math.log(abs(H))
                if TEMP > OFLEXP:
                    raise Exception("Failed to converge")

                HH = math.exp(TEMP)
                scale = lmom_ratios[1] * G * HH / (ALAM2 * GAM)
                loc = lmom_ratios[0] - scale / G * (1 - GAM * U1 / HH)
                return OrderedDict([('k', G),
                                    ('h', H),
                                    ('loc', loc),
                                    ('scale', scale)])
            else:
                XG = G
                XH = H
                XZ = Z
                Xdist = DIST
                RHH = 1 / (H ** 2)
                if H > 0:
                    U1G = -U1 * special.psi(1 / H + 1 + G)
                    U2G = -U2 * special.psi(2 / H + 1 + G)
                    U3G = -U3 * special.psi(3 / H + 1 + G)
                    U4G = -U4 * special.psi(4 / H + 1 + G)
                    U1H = RHH * (-U1G - U1 * special.psi(1 / H))
                    U2H = 2 * RHH * (-U2G - U2 * special.psi(2 / H))
                    U3H = 3 * RHH * (-U3G - U3 * special.psi(3 / H))
                    U4H = 4 * RHH * (-U4G - U4 * special.psi(4 / H))
                else:
                    U1G = -U1 * special.psi(-1 / H - G)
                    U2G = -U2 * special.psi(-2 / H - G)
                    U3G = -U3 * special.psi(-3 / H - G)
                    U4G = -U4 * special.psi(-4 / H - G)
                    U1H = RHH * (-U1G - U1 * special.psi(-1 / H + 1))
                    U2H = 2 * RHH * (-U2G - U2 * special.psi(-2 / H + 1))
                    U3H = 3 * RHH * (-U3G - U3 * special.psi(-3 / H + 1))
                    U4H = 4 * RHH * (-U4G - U4 * special.psi(-4 / H + 1))

                DL2G = U1G - 2 * U2G
                DL2H = U1H - 2 * U2H
                DL3G = -U1G + 6 * U2G - 6 * U3G
                DL3H = -U1H + 6 * U2H - 6 * U3H
                DL4G = U1G - 12 * U2G + 30 * U3G - 20 * U4G
                DL4H = U1H - 12 * U2H + 30 * U3H - 20 * U4H
                D11 = (DL3G - TAU3 * DL2G) / ALAM2
                D12 = (DL3H - TAU3 * DL2H) / ALAM2
                D21 = (DL4G - TAU4 * DL2G) / ALAM2
                D22 = (DL4H - TAU4 * DL2H) / ALAM2
                DET = D11 * D22 - D12 * D21
                H11 = D22 / DET
                H12 = -D12 / DET
                H21 = -D21 / DET
                H22 = D11 / DET
                DEL1 = E1 * H11 + E2 * H12
                DEL2 = E1 * H21 + E2 * H22

                # TAKE NEXT N-R STEP
                G = XG - DEL1
                H = XH - DEL2
                Z = G + H * 0.725

                # REDUCE STEP IF G AND H ARE OUTSIDE THE PARAMETER _spACE

                FACTOR = 1
                if G <= -1:
                    FACTOR = 0.8 * (XG + 1) / DEL1
                if H <= -1:
                    FACTOR = min(FACTOR, 0.8 * (XH + 1) / DEL2)
                if Z <= -1:
                    FACTOR = min(FACTOR, 0.8 * (XZ + 1) / (XZ - Z))
                if H <= 0 and G * H <= -1:
                    FACTOR = min(FACTOR, 0.8 * (XG * XH + 1) / (XG * XH - G * H))

                if FACTOR == 1:
                    pass
                else:
                    DEL1 *= FACTOR
                    DEL2 *= FACTOR
                    G = XG - DEL1
                    H = XH - DEL2
                    Z = G + H * 0.725

    def _lmom_ratios(self, k, h, loc, scale, nmom):
        EU = 0.577215664901532861
        SMALL = 1e-8
        OFL = 170

        if k <= -1 or (h < 0 and (k * h) <= -1):
            raise ValueError("Invalid parameters")

        DLGAM = special.gammaln(1 + k)
        ICASE = 1
        if h > 0:
            ICASE = 3
        if abs(h) < SMALL:
            ICASE = 2
        if k == 0:
            ICASE += 3

        Beta = []
        if ICASE == 1:
            for IR in range(1, nmom + 1):
                ARG = DLGAM + special.gammaln(-IR / h - k) - special.gammaln(-IR / h) - k * math.log(-h)
                if abs(ARG) > OFL:
                    raise Exception("Calculation of L-Moments Failed")
                Beta.append(math.exp(ARG))

        elif ICASE == 2:
            for IR in range(1, nmom + 1):
                Beta.append(math.exp(DLGAM - k * math.log(IR)) * (1 - 0.5 * h * k * (1 + k) / IR))

        elif ICASE == 3:
            for IR in range(1, nmom + 1):
                ARG = DLGAM + special.gammaln(1 + IR / h) - special.gammaln(1 + k + IR / h) - k * math.log(h)
                if abs(ARG) > OFL:
                    raise Exception("Calculation of L-Moments Failed")
                Beta.append(math.exp(ARG))

        elif ICASE == 4:
            for IR in range(1, nmom + 1):
                Beta.append(EU + math.log(-h) + special.psi(-IR / h))

        elif ICASE == 5:
            for IR in range(1, nmom + 1):
                Beta.append(EU + math.log(IR))

        elif ICASE == 6:
            for IR in range(1, nmom + 1):
                Beta.append(EU + math.log(h) + special.psi(1 + IR / h))

        if k == 0:
            xmom = [loc + scale * Beta[0]]
        else:
            xmom = [loc + scale * (1 - Beta[0]) / k]

        if nmom == 1:
            return xmom

        ALAM2 = Beta[1] - Beta[0]
        if k == 0:
            xmom.append(scale * ALAM2)
        else:
            xmom.append(scale * ALAM2 / (-k))

        if nmom == 2:
            return xmom

        Z0 = 1
        for j in range(3, nmom + 1):
            Z0 *= (4.0 * j - 6) / j
            Z = 3 * Z0 * (j - 1) / (j + 1)
            SUMM = Z0 * (Beta[j - 1] - Beta[0]) / ALAM2 - Z
            if j == 3:
                xmom.append(SUMM)
            else:
                for i in range(2, j - 1):
                    Z *= (i + i + 1) * (j - i) / ((i + i - 1) * (j + i))
                    SUMM -= Z * xmom[i]
                xmom.append(SUMM)
        return xmom


kap = KappaGen(name='kappa', shapes='k, h')


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


pe3 = Pearson3Gen(name="pearson3", shapes='skew')


class FrechetRGen(LmomDistrMixin, scipy.stats._continuous_distns.frechet_r_gen):
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


wei = FrechetRGen(a=0.0, name='weibull_min', shapes='c')
