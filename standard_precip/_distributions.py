"""Distribution registry backing the dist_type strings of the public API.

L-moment fits go through the external lmoments3 package (Hosking's algorithms;
the code previously vendored here was an old copy of it). MLE fits go through
plain scipy.stats distributions: calling .fit() on the lmoments3 subclasses
triggers infinite recursion for distributions where scipy's own fit override
uses the legacy super(type(self), self) idiom (e.g. pearson3), and scipy's fit
is what was effectively executed for MLE in earlier versions of this package.

Note that for 'glo', 'gno' and 'kap' the two fit types use different
distribution families: lmoments3 implements Hosking's generalized logistic,
generalized normal and kappa distributions, while MLE uses scipy's
genlogistic, gennorm and kappa4. Parameters from one are not interchangeable
with the other. The CDF is always evaluated with the same distribution object
that produced the parameters, so results are internally consistent.
"""

from dataclasses import dataclass

import scipy.stats as scs
from lmoments3 import distr as lm3_distr


@dataclass(frozen=True)
class DistSpec:
    """One dist_type entry: the objects used for each fit method.

    lmom_dist provides .lmom_fit() returning an OrderedDict of named
    parameters. mle_dist is a plain scipy distribution whose tuple result from
    .fit() is mapped onto mle_param_names (shape names first, then loc/scale,
    in scipy order). Either may be None when that fit type is unsupported.
    """

    lmom_dist: object | None
    mle_dist: object | None
    mle_param_names: tuple | None


DISTRIBUTIONS: dict[str, DistSpec] = {
    'gam': DistSpec(lm3_distr.gam, scs.gamma, ('a', 'loc', 'scale')),
    'exp': DistSpec(lm3_distr.exp, scs.expon, ('loc', 'scale')),
    'gev': DistSpec(lm3_distr.gev, scs.genextreme, ('c', 'loc', 'scale')),
    'gpa': DistSpec(lm3_distr.gpa, scs.genpareto, ('c', 'loc', 'scale')),
    'gum': DistSpec(lm3_distr.gum, scs.gumbel_r, ('loc', 'scale')),
    'nor': DistSpec(lm3_distr.nor, scs.norm, ('loc', 'scale')),
    'pe3': DistSpec(lm3_distr.pe3, scs.pearson3, ('skew', 'loc', 'scale')),
    'wei': DistSpec(lm3_distr.wei, scs.weibull_min, ('c', 'loc', 'scale')),
    'wak': DistSpec(lm3_distr.wak, None, None),
    'glo': DistSpec(lm3_distr.glo, scs.genlogistic, ('c', 'loc', 'scale')),
    'gno': DistSpec(lm3_distr.gno, scs.gennorm, ('beta', 'loc', 'scale')),
    'kap': DistSpec(lm3_distr.kap, scs.kappa4, ('h', 'k', 'loc', 'scale')),
}


def get_spec(dist_type: str) -> DistSpec:
    try:
        return DISTRIBUTIONS[dist_type]
    except KeyError:
        raise ValueError(
            f"'{dist_type}' is not a supported distribution. "
            f"Options are: {', '.join(sorted(DISTRIBUTIONS))}"
        ) from None


def fit(spec: DistSpec, data, fit_type: str, **kwargs):
    """Fit data and return (distribution, params) where params is a dict of
    named parameters accepted by distribution.cdf/.pdf/.ppf."""
    if fit_type == 'lmom':
        if spec.lmom_dist is None:
            raise ValueError("This distribution does not support L-moments fitting")
        return spec.lmom_dist, spec.lmom_dist.lmom_fit(data, **kwargs)

    if fit_type == 'mle':
        if spec.mle_dist is None:
            raise ValueError("This distribution supports L-moments fitting only")
        params = spec.mle_dist.fit(data, **kwargs)
        return spec.mle_dist, dict(zip(spec.mle_param_names, params))

    raise ValueError(f"{fit_type} is not an option. Option fit_types are mle and lmom")


def min_samples(spec: DistSpec, fit_type: str) -> int:
    """Smallest sample size for which fitting is attempted; below this the
    caller should yield NaN rather than fit. lmoments3 requires strictly more
    than numargs + 2 observations."""
    if fit_type == 'lmom' and spec.lmom_dist is not None:
        return max(4, spec.lmom_dist.numargs + 3)
    return 4
