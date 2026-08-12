import numpy as np
import pytest

from standard_precip import spi

INDEX_COL = "TotalPrecipitation_calculated_index"

# Golden values computed with the original vendored l-moments code; they pin the
# numerical behavior of every supported (dist_type, fit_type) combination.
GOLDEN_MONTHLY = [
    # (dist_type, fit_type, dist_kwargs, iloc, expected)
    ("gam", "lmom", {}, 0, -0.678092),
    ("gam", "mle", {"floc": 0}, 0, -0.696543),
    ("exp", "lmom", {}, 0, -0.575136),
    ("exp", "mle", {}, 0, -0.177047),
    ("gev", "lmom", {}, 0, -0.695562),
    ("gev", "mle", {}, 0, -0.701048),
    ("gpa", "lmom", {}, 0, -0.560434),
    ("gpa", "mle", {}, 0, -0.288574),
    ("gum", "lmom", {}, 0, -0.698605),
    ("gum", "mle", {}, 0, -0.720206),
    ("nor", "lmom", {}, 0, -0.753621),
    ("nor", "mle", {}, 0, -0.712801),
    ("pe3", "lmom", {}, 0, -0.652750),
    # pe3 MLE recurses infinitely: scipy's pearson3_gen.fit uses the legacy
    # super(type(self), self) idiom, which breaks when subclassed (as the vendored
    # lmoments code does). Broken with scipy >= 1.7; fixed in the lmoments3 migration.
    pytest.param(
        "pe3", "mle", {}, 0, -0.670490, marks=pytest.mark.xfail(raises=RecursionError)
    ),
    ("wei", "lmom", {}, 0, -0.624138),
    ("wei", "mle", {}, 0, -0.625167),
    ("glo", "mle", {}, 0, -0.721554),
    ("gno", "mle", {}, 0, -0.655203),
    ("kap", "mle", {}, 0, -0.312152),
    ("wak", "lmom", {}, 3, -0.204953),
]


@pytest.mark.parametrize(
    "dist_type, fit_type, dist_kwargs, iloc, expected",
    GOLDEN_MONTHLY,
    ids=[f"{d}-{f}" for d, f, *_ in GOLDEN_MONTHLY],
)
def test_monthly_spi(monthly_df, dist_type, fit_type, dist_kwargs, iloc, expected):
    new_spi = spi.SPI()
    df_spi = new_spi.calculate(
        monthly_df,
        "date",
        "TotalPrecipitation",
        freq="M",
        fit_type=fit_type,
        dist_type=dist_type,
        **dist_kwargs,
    )
    assert df_spi[INDEX_COL].iloc[iloc] == pytest.approx(expected, abs=1e-4)


def test_3month_spi(wichita_df):
    new_spi = spi.SPI()
    df_spi = new_spi.calculate(
        wichita_df, "date", "precip", freq="M", fit_type="lmom", scale=3, dist_type="gam"
    )
    assert df_spi["precip_scale_3_calculated_index"].iloc[2] == pytest.approx(0.856479, abs=1e-4)


def test_daily_nan(daily_test_df):
    new_spi = spi.SPI()
    df_spi = new_spi.calculate(
        daily_test_df, "date", "precip", freq="D", fit_type="lmom", scale=1, dist_type="gam"
    )
    assert np.isnan(df_spi["precip_calculated_index"].iloc[0])
