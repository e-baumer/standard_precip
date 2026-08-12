"""Numerical-equivalence gate.

regression_golden.csv pins the full calculated-index series for every supported
(dist_type, fit_type) combination; these tests assert the package still
reproduces it to near machine precision. The file was originally generated
(tests/gen_golden.py) against the vendored l-moments code just before the
migration to the external lmoments3 package, proving equivalence across that
swap, and has been regenerated once since: out-of-support observations now map
to large finite index values instead of NaN (CDF clipped at 1e-16), a change
that affected only previously-NaN cells.

A failure here means the numerical output of the package changed. Never accept
that silently: regenerating the golden is legitimate only for an intentional,
CHANGELOG-documented numerical change whose diff against the old golden has
been inspected and explained (as above) - not to make a red gate green.
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
