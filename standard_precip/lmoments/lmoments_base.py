# -*- coding: utf-8 -*-

"""
This file contains a Python implimentation of the lmoments.f library created by
J. R. M. HOSKING.
The base Fortran code is copyright of the IBM Corperation, and the licensing
information is shown below:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IBM software disclaimer
LMOMENTS: Fortran routines for use with the method of L-moments
Permission to use, copy, modify and distribute this software for any purpose
and without fee is hereby granted, provided that this copyright and permission
notice appear on all copies of the software. The name of the IBM Corporation
may not be used in any advertising or publicity pertaining to the use of the
software. IBM makes no warranty or representations about the suitability of the
software for any purpose. It is provided "AS IS" without any express or implied
warranty, including the implied warranties of merchantability, fitness for a
particular purpose and non-infringement. IBM shall not be liable for any direct,
indirect, _special or consequential damages resulting from the loss of use,
data or projects, whether in an action of contract or tort, arising out of or
in connection with the use or performance of this software.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Additional code from the R library "lmomco" has been converted into Python.
This library was developed by WILLIAM ASQUITH, and was released under the GPL-3
License. Copyright (C) 2012 WILLIAM ASQUITH
The Python translation was conducted by:
    Sam Gillespie
    Numerical Analyst
    C&R Consulting
    Townsville Australia
    September 2013
For more information, or to report bugs, contact:
    sam@candrconsulting.com.au
Licensing for Python Translation:
####################################################
    Copyright (C) 2014 Sam Gillespie
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.Version 0.1.0:
####################################################
"""
import scipy.special
import numpy as np


def lmom_ratios_fn(data, nmom=5):
    """
    Estimate `nmom` number of L-moments from a sample `data`.
    :param data: Sequence of (sample) data
    :type data: list or array-like sequence
    :param nmom: number of L-moments to estimate
    :type nmom: int
    :return: L-moment ratios like this: l1, l2, t3, t4, t5, .. . As in: items 3 and higher are L-moment ratios.
    :rtype: list
    """

    if nmom <= 5:
        return _samlmusmall(data, nmom)
    else:
        return _samlmularge(data, nmom)


def _samlmularge(x, nmom=5):
    try:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        x.sort()
    except ValueError:
        raise ValueError("Input data to estimate L-moments must be numeric.")

    if nmom <= 0:
        raise ValueError("Invalid number of sample L-moments")

    if n < nmom:
        raise ValueError("Insufficient length of data for specified nmoments")

    ##Calculate first order
    l = [np.sum(x) / scipy.special.comb(n, 1, exact=True)]

    if nmom == 1:
        return l[0]

    #Setup comb table, where comb[i][x] refers to comb(x,i)
    comb = []
    for i in range(1, nmom):
        comb.append([])
        for j in range(n):
            comb[-1].append(scipy.special.comb(j, i, exact=True))

    for mom in range(2, nmom + 1):
        coefl = 1.0 / mom * 1.0 / scipy.special.comb(n, mom, exact=True)
        xtrans = []
        for i in range(0, n):
            coeftemp = []
            for j in range(0, mom):
                coeftemp.append(1)

            for j in range(0, mom - 1):
                coeftemp[j] = coeftemp[j] * comb[mom - j - 2][i]

            for j in range(1, mom):
                coeftemp[j] = coeftemp[j] * comb[j - 1][n - i - 1]

            for j in range(0, mom):
                coeftemp[j] = coeftemp[j] * scipy.special.comb(mom - 1, j, exact=True)

            for j in range(0, int(0.5 * mom)):
                coeftemp[j * 2 + 1] = -coeftemp[j * 2 + 1]
            coeftemp = sum(coeftemp)
            xtrans.append(x[i] * coeftemp)

        if mom > 2:
            l.append(coefl * sum(xtrans) / l[1])
        else:
            l.append(coefl * sum(xtrans))
    return l


def _samlmusmall(x, nmom=5):
    try:
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        x.sort()
    except ValueError:
        raise ValueError("Input data to estimate L-moments must be numeric.")

    if nmom <= 0 or nmom > 5:
        raise ValueError("Invalid number of sample L-moments")

    if n < nmom:
        raise ValueError("Insufficient length of data for specified nmoments")

    # First L-moment

    l1 = np.sum(x) / scipy.special.comb(n, 1, exact=True)

    if nmom == 1:
        return l1

    # Second L-moment

    comb1 = range(n)
    coefl2 = 0.5 / scipy.special.comb(n, 2, exact=True)
    sum_xtrans = sum([(comb1[i] - comb1[n - i - 1]) * x[i] for i in range(n)])
    l2 = coefl2 * sum_xtrans

    if nmom == 2:
        return [l1, l2]

    # Third L-moment

    comb3 = [scipy.special.comb(i, 2, exact=True) for i in range(n)]
    coefl3 = 1.0 / 3.0 / scipy.special.comb(n, 3, exact=True)
    sum_xtrans = sum([(comb3[i] - 2 * comb1[i] * comb1[n - i - 1] + comb3[n - i - 1]) * x[i] for i in range(n)])
    l3 = coefl3 * sum_xtrans / l2

    if nmom == 3:
        return [l1, l2, l3]

    # Fourth L-moment

    comb5 = [scipy.special.comb(i, 3, exact=True) for i in range(n)]
    coefl4 = 0.25 / scipy.special.comb(n, 4, exact=True)
    sum_xtrans = sum(
        [(comb5[i] - 3 * comb3[i] * comb1[n - i - 1] + 3 * comb1[i] * comb3[n - i - 1] - comb5[n - i - 1]) * x[i]
         for i in range(n)])
    l4 = coefl4 * sum_xtrans / l2

    if nmom == 4:
        return [l1, l2, l3, l4]

    # Fifth L-moment

    comb7 = [scipy.special.comb(i, 4, exact=True) for i in range(n)]
    coefl5 = 0.2 / scipy.special.comb(n, 5, exact=True)
    sum_xtrans = sum(
        [(comb7[i] - 4 * comb5[i] * comb1[n - i - 1] + 6 * comb3[i] * comb3[n - i - 1] -
          4 * comb1[i] * comb5[n - i - 1] + comb7[n - i - 1]) * x[i]
         for i in range(n)])
    l5 = coefl5 * sum_xtrans / l2

    return [l1, l2, l3, l4, l5]
