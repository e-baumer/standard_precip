"""Tests for the local Wakeby distribution (standard_precip._wakeby).

lmoments3's own Wakeby CDF is wrong for fitted parameters (it scales gamma by
alpha a second time and returns NaN whenever delta < 0 or alpha = 0), so these
tests check the local implementation against references that do not depend on
it: Hosking's quantile formula written out directly, the sample L-moments the
fit must reproduce, scipy's generalized Pareto and exponential special cases,
and lmoments3 itself in the one regime where it is right (scale = 1 with
delta > 0).
"""

import lmoments3
import numpy as np
import pandas as pd
import pytest
import scipy.stats as scs
from lmoments3 import distr as lm3_distr
from scipy.integrate import quad

from standard_precip import _distributions
from standard_precip._wakeby import wak
from standard_precip.spi import SPI

PARAM_SETS = {
    "unit-scale": dict(beta=2.0, gamma=0.5, delta=0.2, loc=0.0, scale=1.0),
    "scaled-positive-delta": dict(beta=2.7, gamma=0.62, delta=0.19, loc=0.4, scale=4.45),
    "negative-delta": dict(beta=5.7, gamma=0.96, delta=-0.05, loc=0.3, scale=3.5),
    "zero-delta": dict(beta=3.0, gamma=1.2, delta=0.0, loc=0.0, scale=2.0),
    "gpa-fallback": dict(beta=0.0, gamma=2.0, delta=0.1, loc=1.0, scale=0.0),
    "gamma-zero": dict(beta=2.0, gamma=0.0, delta=0.0, loc=0.0, scale=5.0),
    "exponential": dict(beta=0.0, gamma=0.0, delta=0.0, loc=2.0, scale=3.0),
}

all_param_sets = pytest.mark.parametrize("params", PARAM_SETS.values(), ids=PARAM_SETS.keys())


def hosking_quantile(f, beta, gamma, delta, loc, scale):
    """Hosking and Wallis (1997) eq. A.98 written with plain powers, valid for
    nonzero beta and delta."""
    return loc + scale / beta * (1 - (1 - f) ** beta) - gamma / delta * (1 - (1 - f) ** (-delta))


@pytest.mark.parametrize("name", ["unit-scale", "scaled-positive-delta", "negative-delta"])
def test_ppf_matches_hosking_formula(name):
    params = PARAM_SETS[name]
    q = np.linspace(0.001, 0.999, 201)
    np.testing.assert_allclose(wak.ppf(q, **params), hosking_quantile(q, **params), rtol=1e-12)


@all_param_sets
def test_cdf_inverts_ppf(params):
    q = np.linspace(0.001, 0.999, 201)
    np.testing.assert_allclose(wak.cdf(wak.ppf(q, **params), **params), q, atol=1e-12)


@all_param_sets
def test_pdf_integrates_to_cdf(params):
    upper = float(wak.ppf(0.9, **params))
    area, _ = quad(lambda x: float(wak.pdf(x, **params)), params["loc"], upper, limit=200)
    assert area == pytest.approx(0.9, abs=1e-7)


@pytest.mark.parametrize("seed", range(5))
def test_fit_reproduces_sample_l_moments(seed):
    """A five-parameter Wakeby fit matches the first L-moments of the sample,
    so integrating the fitted quantile function must return them. Scaling
    gamma by alpha a second time breaks this."""
    data = np.random.default_rng(seed).gamma(2.0, 30.0, 300)
    params = wak.lmom_fit(data)
    l1, l2 = lmoments3.lmom_ratios(data, nmom=2)
    fitted_l1, _ = quad(lambda f: float(wak.ppf(f, **params)), 0, 1, limit=200)
    fitted_l2, _ = quad(lambda f: float(wak.ppf(f, **params)) * (2 * f - 1), 0, 1, limit=200)
    assert fitted_l1 == pytest.approx(l1, rel=1e-7)
    assert fitted_l2 == pytest.approx(l2, rel=1e-7)


@pytest.mark.parametrize("shapes", [(2.0, 0.5, 0.2), (0.5, 1.5, 0.05), (5.0, 0.1, 0.4)])
def test_matches_lmoments3_at_unit_scale(shapes):
    x = wak.ppf(np.linspace(0.01, 0.99, 99), *shapes)
    np.testing.assert_allclose(wak.cdf(x, *shapes), lm3_distr.wak.cdf(x, *shapes), atol=1e-7)


def test_gpa_fallback_matches_genpareto():
    x = np.linspace(1.0, 40.0, 50)
    expected = scs.genpareto.cdf(x, 0.1, loc=1.0, scale=2.0)
    np.testing.assert_allclose(wak.cdf(x, **PARAM_SETS["gpa-fallback"]), expected, atol=1e-12)


def test_exponential_special_case():
    x = np.linspace(2.0, 30.0, 50)
    expected = scs.expon.cdf(x, loc=2.0, scale=3.0)
    np.testing.assert_allclose(wak.cdf(x, **PARAM_SETS["exponential"]), expected, atol=1e-12)


def test_support_bounds_and_nan():
    params = PARAM_SETS["negative-delta"]
    upper = params["loc"] + params["scale"] / params["beta"] - params["gamma"] / params["delta"]
    assert wak.ppf(1.0, **params) == pytest.approx(upper)
    cdf = wak.cdf([params["loc"] - 1.0, params["loc"], upper, upper + 1.0, np.nan], **params)
    np.testing.assert_array_equal(cdf[:4], [0.0, 0.0, 1.0, 1.0])
    assert np.isnan(cdf[4])
    assert wak.pdf(upper + 1.0, **params) == 0.0
    assert wak.pdf(params["loc"] - 1.0, **params) == 0.0


def test_unbounded_upper_tail():
    assert np.isinf(wak.ppf(1.0, **PARAM_SETS["unit-scale"]))
    assert np.isinf(wak.ppf(1.0, **PARAM_SETS["zero-delta"]))
    assert wak.ppf(0.0, **PARAM_SETS["unit-scale"]) == 0.0


@pytest.mark.parametrize(
    "shapes",
    [(1.0, -1.0, 0.5), (1.0, 0.5, 1.2), (-1.0, 0.5, 0.5), (1.0, 0.0, 0.5)],
    ids=["negative-gamma", "delta-ge-1", "beta-plus-delta-le-0", "delta-without-gamma"],
)
def test_invalid_parameters_give_nan(shapes):
    assert np.isnan(wak.cdf([1.0], *shapes)).all()
    assert np.isnan(wak.pdf([1.0], *shapes)).all()
    assert np.isnan(wak.ppf([0.5], *shapes)).all()


def test_ppf_outside_unit_interval_is_nan():
    out = wak.ppf([-0.1, 0.5, 1.1], **PARAM_SETS["unit-scale"])
    assert np.isnan(out[0]) and np.isnan(out[2]) and np.isfinite(out[1])


def test_registry_uses_local_wakeby():
    assert _distributions.get_spec("wak").lmom_dist is wak


def test_wak_spi_is_finite_on_monthly_data(monthly_df):
    """Before the fix 9 of the 12 calendar months of this record were entirely
    NaN, because their fitted delta is negative."""
    result = SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="M", dist_type="wak")
    deltas = [
        wak.lmom_fit(group["TotalPrecipitation"].to_numpy())["delta"]
        for _, group in monthly_df.groupby(pd.to_datetime(monthly_df["date"]).dt.month)
    ]
    assert min(deltas) < 0 < max(deltas)
    assert np.isfinite(result["TotalPrecipitation_calculated_index"]).all()


def test_wak_spi3_agrees_with_gamma_on_gamma_data():
    """SPI-3 on 60 years of gamma-distributed monthly totals. The Wakeby index
    must be finite, close to standard normal, and close to the index from a
    gamma fit of the same data."""
    rng = np.random.default_rng(35)
    df = pd.DataFrame(
        {
            "date": pd.date_range("1960-01-01", periods=720, freq="MS"),
            "p": rng.gamma(2.0, 30.0, 720),
        }
    )
    col = "p_scale_3_calculated_index"
    by_wak = SPI().calculate(df, "date", "p", freq="M", scale=3, dist_type="wak")[col].iloc[2:]
    by_gam = SPI().calculate(df, "date", "p", freq="M", scale=3, dist_type="gam")[col].iloc[2:]

    assert np.isfinite(by_wak).all()
    in_support = by_wak.abs() < 8
    assert in_support.mean() > 0.97
    assert by_wak[in_support].mean() == pytest.approx(0.0, abs=0.1)
    assert by_wak[in_support].std() == pytest.approx(1.0, abs=0.1)
    assert (by_wak < -1.5).mean() == pytest.approx(scs.norm.cdf(-1.5), abs=0.03)
    assert np.corrcoef(by_wak[in_support], by_gam[in_support])[0, 1] > 0.98
