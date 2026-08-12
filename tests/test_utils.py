import numpy as np
import pytest

from standard_precip.spi import SPI
from standard_precip.utils import best_fit_distribution, plot_index


@pytest.fixture
def gamma_sample():
    rng = np.random.default_rng(11)
    return rng.gamma(shape=2.0, scale=3.0, size=300)


def test_best_fit_distribution_lmom(gamma_sample, tmp_path):
    sse = best_fit_distribution(
        gamma_sample,
        ["gam", "nor", "gum"],
        fit_type="lmom",
        save_file=str(tmp_path / "fits.png"),
    )
    assert (tmp_path / "fits.png").exists()
    dists = [name for name, _ in sse]
    errors = [err for _, err in sse]
    assert sorted(errors) == errors
    assert set(dists) == {"gam", "nor", "gum"}
    assert dists[0] == "gam"


def test_best_fit_distribution_mle(gamma_sample, tmp_path):
    sse = best_fit_distribution(
        gamma_sample,
        ["nor", "gum"],
        fit_type="mle",
        save_file=str(tmp_path / "fits.png"),
    )
    assert len(sse) == 2


def test_plot_index(monthly_df, tmp_path):
    df_spi = SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="M")
    df_spi["date"] = np.array(df_spi["date"], dtype="datetime64[ns]")
    fig = plot_index(
        df_spi,
        "date",
        "TotalPrecipitation_calculated_index",
        save_file=str(tmp_path / "index.png"),
    )
    assert fig is not None
    assert (tmp_path / "index.png").exists()


def test_gam_mle_defaults_to_floc_zero(monthly_df):
    default = SPI().calculate(
        monthly_df, "date", "TotalPrecipitation", freq="M", fit_type="mle", dist_type="gam"
    )
    explicit = SPI().calculate(
        monthly_df,
        "date",
        "TotalPrecipitation",
        freq="M",
        fit_type="mle",
        dist_type="gam",
        floc=0,
    )
    np.testing.assert_allclose(
        default["TotalPrecipitation_calculated_index"].values,
        explicit["TotalPrecipitation_calculated_index"].values,
    )
