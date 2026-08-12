import numpy as np
import pandas as pd
import pytest

from standard_precip.spi import SPI

INDEX_COL = "TotalPrecipitation_calculated_index"


class TestBaselinePeriod:
    def test_full_record_baseline_equals_default(self, monthly_df):
        default = SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="M")
        explicit = SPI().calculate(
            monthly_df, "date", "TotalPrecipitation", freq="M",
            baseline_start=1893, baseline_end=2020,
        )
        np.testing.assert_allclose(default[INDEX_COL].values, explicit[INDEX_COL].values)

    def test_baseline_subset_changes_index(self, monthly_df):
        default = SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="M")
        baseline = SPI().calculate(
            monthly_df, "date", "TotalPrecipitation", freq="M",
            baseline_start=1961, baseline_end=1990,
        )
        # The full record is transformed in both cases...
        assert baseline[INDEX_COL].notna().sum() == default[INDEX_COL].notna().sum()
        # ...but relative to a different reference climatology.
        assert not np.allclose(
            default[INDEX_COL].dropna().values, baseline[INDEX_COL].dropna().values
        )

    def test_baseline_mean_is_zeroish_within_baseline(self, monthly_df):
        baseline = SPI().calculate(
            monthly_df, "date", "TotalPrecipitation", freq="M",
            baseline_start=1961, baseline_end=1990,
        )
        dates = pd.to_datetime(baseline["date"])
        inside = baseline.loc[dates.dt.year.between(1961, 1990), INDEX_COL]
        assert abs(inside.mean()) < 0.1

    def test_string_bounds(self, monthly_df):
        by_year = SPI().calculate(
            monthly_df, "date", "TotalPrecipitation", freq="M",
            baseline_start=1961, baseline_end=1990,
        )
        by_date = SPI().calculate(
            monthly_df, "date", "TotalPrecipitation", freq="M",
            baseline_start="1961-01-01", baseline_end="1990-12-31",
        )
        np.testing.assert_allclose(by_year[INDEX_COL].values, by_date[INDEX_COL].values)

    def test_one_sided_baseline_raises(self, monthly_df):
        with pytest.raises(ValueError, match="together"):
            SPI().calculate(
                monthly_df, "date", "TotalPrecipitation", freq="M", baseline_start=1961
            )

    def test_inverted_baseline_raises(self, monthly_df):
        with pytest.raises(ValueError, match="after"):
            SPI().calculate(
                monthly_df, "date", "TotalPrecipitation", freq="M",
                baseline_start=1990, baseline_end=1961,
            )

    def test_empty_baseline_raises(self, monthly_df):
        with pytest.raises(ValueError, match="No observations"):
            SPI().calculate(
                monthly_df, "date", "TotalPrecipitation", freq="M",
                baseline_start=1700, baseline_end=1750,
            )


class TestReturnParams:
    def test_shape_and_columns(self, monthly_df):
        df_spi, df_params = SPI().calculate(
            monthly_df, "date", "TotalPrecipitation", freq="M", return_params=True
        )
        assert len(df_params) == 12  # one row per month
        for col in ["column", "freq_group", "dist_type", "fit_type", "n_fit", "p_zero"]:
            assert col in df_params.columns
        # gamma fit parameters
        assert {"a", "loc", "scale"} <= set(df_params.columns)
        assert (df_params["column"] == "TotalPrecipitation").all()
        assert sorted(df_params["freq_group"]) == list(range(1, 13))
        assert df_params["p_zero"].between(0, 1).all()

    def test_index_result_unchanged(self, monthly_df):
        alone = SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="M")
        both, _ = SPI().calculate(
            monthly_df, "date", "TotalPrecipitation", freq="M", return_params=True
        )
        np.testing.assert_allclose(alone[INDEX_COL].values, both[INDEX_COL].values)

    def test_p_zero_none_for_unbounded_dist(self, monthly_df):
        _, df_params = SPI().calculate(
            monthly_df, "date", "TotalPrecipitation", freq="M",
            dist_type="nor", return_params=True,
        )
        assert df_params["p_zero"].isna().all()
