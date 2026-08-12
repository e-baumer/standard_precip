import dataclasses

import numpy as np
import pandas as pd
import pytest

from standard_precip import _distributions
from standard_precip.spi import SPI
from standard_precip.utils import best_fit_distribution


@pytest.fixture
def zero_inflated_monthly():
    rng = np.random.default_rng(10)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2000-01-01", periods=120, freq="MS"),
            "p": rng.gamma(2, 3, 120),
        }
    )
    jan_rows = df.index[pd.to_datetime(df["date"]).dt.month == 1]
    df.loc[jan_rows[:3], "p"] = 0.0
    return df


def test_gam_mle_default_applied_in_registry():
    rng = np.random.default_rng(11)
    data = rng.gamma(2.0, 3.0, 200)
    spec = _distributions.get_spec("gam")
    _, params = _distributions.fit(spec, data, "mle")
    assert params["loc"] == 0
    _, params_free = _distributions.fit(spec, data, "mle", loc=0.1)
    assert params_free["loc"] != 0


def test_specs_are_immutable():
    spec = _distributions.get_spec("gam")
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.strip_zeros = False
    with pytest.raises(TypeError):
        spec.mle_defaults["floc"] = 1


def test_strip_zeros_flags():
    stripped = {name for name, s in _distributions.DISTRIBUTIONS.items() if s.strip_zeros}
    assert stripped == {"gam", "pe3"}


def test_n_fit_excludes_stripped_zeros(zero_inflated_monthly):
    _, df_params = SPI().calculate(zero_inflated_monthly, "date", "p", freq="M", return_params=True)
    jan = df_params[df_params["freq_group"] == 1].iloc[0]
    assert jan["n_fit"] == 7
    assert jan["p_zero"] == pytest.approx(0.3)


def test_best_fit_matches_calculate_model(zero_inflated_monthly, tmp_path):
    data = zero_inflated_monthly["p"].values
    sse = best_fit_distribution(
        data, ["gam", "nor"], fit_type="mle", save_file=str(tmp_path / "f.png")
    )
    assert {name for name, _ in sse} == {"gam", "nor"}
    assert all(np.isfinite(err) for _, err in sse)
