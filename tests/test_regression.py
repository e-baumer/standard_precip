"""Numerical-equivalence gate.

regression_golden.csv was generated (tests/gen_golden.py) against the vendored
l-moments code just before the migration to the external lmoments3 package.
These tests assert that every supported (dist_type, fit_type) combination still
reproduces that behavior to near machine precision. A failure here means the
numerical output of the package changed - never accept that silently.
"""

from pathlib import Path

import pandas as pd
import pytest

from standard_precip.spi import SPI

GOLDEN_CSV = Path(__file__).resolve().parent / "regression_golden.csv"


def _load_golden():
    golden = pd.read_csv(GOLDEN_CSV)
    return {
        (dist_type, fit_type): group
        for (dist_type, fit_type), group in golden.groupby(["dist_type", "fit_type"])
    }

GOLDEN = _load_golden()
FIT_KWARGS = {("gam", "mle"): {"floc": 0}, ("wei", "mle"): {"floc": 0}}


@pytest.mark.parametrize("dist_type, fit_type", GOLDEN, ids=lambda v: v)
def test_numerical_equivalence(monthly_df, dist_type, fit_type):
    expected = GOLDEN[(dist_type, fit_type)]
    result = SPI().calculate(
        monthly_df,
        "date",
        "TotalPrecipitation",
        freq="M",
        fit_type=fit_type,
        dist_type=dist_type,
        **FIT_KWARGS.get((dist_type, fit_type), {}),
    )
    assert result["date"].astype(str).tolist() == expected["date"].tolist()
    assert result["TotalPrecipitation_calculated_index"].values == pytest.approx(
        expected["index_value"].values, rel=1e-9, nan_ok=True
    )
