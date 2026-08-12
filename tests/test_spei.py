import numpy as np
import pandas as pd
import pytest

from standard_precip import SPEI


@pytest.fixture
def water_balance_df():
    rng = np.random.default_rng(123)
    dates = pd.date_range("1970-01-01", periods=600, freq="MS")
    seasonal = 30 * np.sin(2 * np.pi * (dates.month - 1) / 12)
    d = seasonal + rng.normal(loc=-5, scale=25, size=len(dates))
    return pd.DataFrame({"date": dates, "wb": d})


def test_spei_glo_lmom(water_balance_df):
    assert (water_balance_df["wb"] < 0).any()
    out = SPEI().calculate(
        water_balance_df, "date", "wb", freq="M", dist_type="glo", fit_type="lmom"
    )
    idx = out["wb_calculated_index"]
    assert idx.notna().all()
    assert abs(idx.mean()) < 0.1
    assert 0.8 < idx.std() < 1.2


def test_spei_negative_values_bypass_p_zero(water_balance_df):
    _, params = SPEI().calculate(
        water_balance_df,
        "date",
        "wb",
        freq="M",
        dist_type="glo",
        fit_type="lmom",
        return_params=True,
    )
    assert params["p_zero"].isna().all()


def test_spei_golden(water_balance_df):
    out = SPEI().calculate(
        water_balance_df, "date", "wb", freq="M", dist_type="glo", fit_type="lmom"
    )
    assert out["wb_calculated_index"].iloc[0] == pytest.approx(-1.002073, abs=1e-4)
