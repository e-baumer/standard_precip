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


class TestAnnualMode:
    def test_annual_spi(self):
        # 60 years of synthetic annual precipitation totals: with freq=None all
        # years form one fitting population, so the index should be ~N(0, 1).
        rng = np.random.default_rng(42)
        df = pd.DataFrame(
            {
                "date": pd.date_range("1950-12-31", periods=60, freq="YE"),
                "precip": rng.gamma(shape=20.0, scale=40.0, size=60),
            }
        )
        out = SPI().calculate(df, "date", "precip", freq=None)
        idx = out["precip_calculated_index"]
        assert idx.notna().all()
        assert abs(idx.mean()) < 0.15
        assert 0.8 < idx.std() < 1.2

    def test_invalid_freq_raises(self, monthly_df):
        with pytest.raises(ValueError, match="not a recognized frequency"):
            SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="Y")

    def test_freq_col_equivalent_to_month(self, monthly_df):
        by_freq = SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="M")
        df = monthly_df.copy()
        df["month_group"] = pd.to_datetime(df["date"]).dt.month
        by_col = SPI().calculate(
            df, "date", "TotalPrecipitation", freq_col="month_group"
        )
        np.testing.assert_allclose(by_freq[INDEX_COL].values, by_col[INDEX_COL].values)

    def test_missing_freq_col_raises(self, monthly_df):
        with pytest.raises(ValueError, match="not a column"):
            SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq_col="nope")

    def test_non_integer_freq_col_raises(self, monthly_df):
        df = monthly_df.copy()
        df["bad_group"] = "january"
        with pytest.raises(ValueError, match="integer column"):
            SPI().calculate(monthly_df.assign(bad_group="x"), "date", "TotalPrecipitation",
                            freq_col="bad_group")


class TestMultiColumn:
    def test_two_columns(self, monthly_df):
        df = monthly_df.copy()
        rng = np.random.default_rng(7)
        df["precip2"] = df["TotalPrecipitation"] * rng.uniform(0.5, 1.5, len(df))
        out = SPI().calculate(df, "date", ["TotalPrecipitation", "precip2"], freq="M")
        assert INDEX_COL in out.columns
        assert "precip2_calculated_index" in out.columns
        single = SPI().calculate(df, "date", "TotalPrecipitation", freq="M")
        np.testing.assert_allclose(out[INDEX_COL].values, single[INDEX_COL].values)


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
