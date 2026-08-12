import numpy as np
import pandas as pd
import pytest

from standard_precip.spi import SPI

INDEX_COL = "TotalPrecipitation_calculated_index"


class TestBaselinePeriod:
    def test_full_record_baseline_equals_default(self, monthly_df):
        default = SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="M")
        explicit = SPI().calculate(
            monthly_df,
            "date",
            "TotalPrecipitation",
            freq="M",
            baseline_start=1893,
            baseline_end=2020,
        )
        np.testing.assert_allclose(default[INDEX_COL].values, explicit[INDEX_COL].values)

    def test_baseline_subset_changes_index(self, monthly_df):
        default = SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="M")
        baseline = SPI().calculate(
            monthly_df,
            "date",
            "TotalPrecipitation",
            freq="M",
            baseline_start=1961,
            baseline_end=1990,
        )
        assert baseline[INDEX_COL].notna().sum() == default[INDEX_COL].notna().sum()
        assert not np.allclose(
            default[INDEX_COL].dropna().values, baseline[INDEX_COL].dropna().values
        )

    def test_baseline_mean_is_zeroish_within_baseline(self, monthly_df):
        baseline = SPI().calculate(
            monthly_df,
            "date",
            "TotalPrecipitation",
            freq="M",
            baseline_start=1961,
            baseline_end=1990,
        )
        dates = pd.to_datetime(baseline["date"])
        inside = baseline.loc[dates.dt.year.between(1961, 1990), INDEX_COL]
        assert abs(inside.mean()) < 0.1

    def test_string_bounds(self, monthly_df):
        by_year = SPI().calculate(
            monthly_df,
            "date",
            "TotalPrecipitation",
            freq="M",
            baseline_start=1961,
            baseline_end=1990,
        )
        by_date = SPI().calculate(
            monthly_df,
            "date",
            "TotalPrecipitation",
            freq="M",
            baseline_start="1961-01-01",
            baseline_end="1990-12-31",
        )
        np.testing.assert_allclose(by_year[INDEX_COL].values, by_date[INDEX_COL].values)

    def test_one_sided_baseline_raises(self, monthly_df):
        with pytest.raises(ValueError, match="together"):
            SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="M", baseline_start=1961)

    def test_inverted_baseline_raises(self, monthly_df):
        with pytest.raises(ValueError, match="after"):
            SPI().calculate(
                monthly_df,
                "date",
                "TotalPrecipitation",
                freq="M",
                baseline_start=1990,
                baseline_end=1961,
            )

    def test_empty_baseline_raises(self, monthly_df):
        with pytest.raises(ValueError, match="No observations"):
            SPI().calculate(
                monthly_df,
                "date",
                "TotalPrecipitation",
                freq="M",
                baseline_start=1700,
                baseline_end=1750,
            )


class TestAnnualMode:
    def test_annual_spi(self):
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
        by_col = SPI().calculate(df, "date", "TotalPrecipitation", freq_col="month_group")
        np.testing.assert_allclose(by_freq[INDEX_COL].values, by_col[INDEX_COL].values)

    def test_missing_freq_col_raises(self, monthly_df):
        with pytest.raises(ValueError, match="not a column"):
            SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq_col="nope")

    def test_non_integer_freq_col_raises(self, monthly_df):
        df = monthly_df.assign(bad_group="january")
        with pytest.raises(ValueError, match="integer column"):
            SPI().calculate(df, "date", "TotalPrecipitation", freq_col="bad_group")


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


class TestRobustness:
    def test_unsorted_input_equals_sorted(self):
        rng = np.random.default_rng(3)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2000-01-01", periods=120, freq="MS"),
                "p": rng.gamma(2, 3, 120),
            }
        )
        shuffled = df.sample(frac=1, random_state=0).reset_index(drop=True)
        col = "p_scale_3_calculated_index"
        out_sorted = SPI().calculate(df, "date", "p", scale=3)
        out_shuffled = SPI().calculate(shuffled, "date", "p", scale=3)
        np.testing.assert_allclose(out_sorted[col].values, out_shuffled[col].values)
        assert out_sorted["date"].is_monotonic_increasing

    def test_irregular_spacing_warns_with_scale(self):
        rng = np.random.default_rng(4)
        dates = pd.date_range("2000-01-01", periods=121, freq="MS").delete(60)
        df = pd.DataFrame({"date": dates, "p": rng.gamma(2, 3, 120)})
        with pytest.warns(UserWarning, match="not regularly spaced"):
            SPI().calculate(df, "date", "p", scale=3)

    def test_regular_spacing_does_not_warn(self, recwarn):
        rng = np.random.default_rng(5)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2000-01-01", periods=120, freq="MS"),
                "p": rng.gamma(2, 3, 120),
            }
        )
        SPI().calculate(df, "date", "p", scale=3)
        assert not [w for w in recwarn if "not regularly spaced" in str(w.message)]

    def test_nan_in_one_column_does_not_affect_other(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2000-01-01", periods=120, freq="MS"),
                "a": rng.gamma(2, 3, 120),
                "b": rng.gamma(2, 3, 120),
            }
        )
        df.loc[5, "b"] = np.nan
        both = SPI().calculate(df, "date", ["a", "b"])
        alone = SPI().calculate(df[["date", "a"]], "date", "a")
        np.testing.assert_allclose(
            both["a_calculated_index"].values, alone["a_calculated_index"].values
        )
        assert np.isnan(both["b_calculated_index"].iloc[5])

    def test_out_of_baseline_extreme_is_finite(self):
        rng = np.random.default_rng(2)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2000-01-01", periods=120, freq="MS"),
                "p": rng.gamma(2, 3, 120),
            }
        )
        df.loc[119, "p"] = 5000.0
        out = SPI().calculate(df, "date", "p", baseline_start=2000, baseline_end=2008)
        extreme = out["p_calculated_index"].iloc[119]
        assert np.isfinite(extreme)
        assert 4 < extreme < 9

    def test_partial_year_baseline_with_empty_groups(self):
        rng = np.random.default_rng(6)
        df = pd.DataFrame(
            {
                "date": pd.date_range("1970-01-01", "1972-12-31", freq="D"),
                "p": rng.gamma(2, 3, 1096),
            }
        )
        with pytest.warns(UserWarning, match="Could not fit"):
            out = SPI().calculate(
                df,
                "date",
                "p",
                freq="W",
                baseline_start="1971-01-01",
                baseline_end="1971-06-30",
            )
        dates = pd.to_datetime(out["date"])
        weeks = dates.dt.isocalendar().week
        idx = out["p_calculated_index"]
        assert idx[weeks <= 20].notna().all()
        assert idx[(weeks >= 30) & (weeks <= 50)].isna().all()

    def test_unconvergeable_group_yields_nan_not_crash(self):
        rng = np.random.default_rng(28)
        df = pd.DataFrame(
            {
                "date": pd.date_range("2000-12-31", periods=12, freq=pd.DateOffset(years=1)),
                "p": rng.gamma(0.3, 5, 12),
            }
        )
        with pytest.warns(UserWarning, match="Could not fit 'kap'"):
            out = SPI().calculate(df, "date", "p", freq=None, dist_type="kap", fit_type="lmom")
        assert out["p_calculated_index"].isna().all()

    def test_duplicate_index_labels_with_baseline(self):
        rng = np.random.default_rng(9)
        dates = pd.date_range("2000-01-01", periods=120, freq="MS")
        half_a = pd.DataFrame({"date": dates[:60], "p": rng.gamma(2, 3, 60)})
        half_b = pd.DataFrame({"date": dates[60:], "p": rng.gamma(2, 3, 60)})
        stacked = pd.concat([half_a, half_b])
        assert stacked.index.has_duplicates
        clean = pd.concat([half_a, half_b], ignore_index=True)
        out_dup = SPI().calculate(stacked, "date", "p", baseline_start=2000, baseline_end=2005)
        out_clean = SPI().calculate(clean, "date", "p", baseline_start=2000, baseline_end=2005)
        np.testing.assert_allclose(
            out_dup["p_calculated_index"].values, out_clean["p_calculated_index"].values
        )

    def test_numpy_integer_baseline_years(self, monthly_df):
        by_int = SPI().calculate(
            monthly_df, "date", "TotalPrecipitation", baseline_start=1961, baseline_end=1990
        )
        by_np = SPI().calculate(
            monthly_df,
            "date",
            "TotalPrecipitation",
            baseline_start=np.int64(1961),
            baseline_end=np.int64(1990),
        )
        np.testing.assert_allclose(by_int[INDEX_COL].values, by_np[INDEX_COL].values)

    def test_sparse_data_warns_once_per_column(self, daily_test_df, recwarn):
        SPI().calculate(daily_test_df, "date", "precip", freq="D")
        fit_warnings = [w for w in recwarn if "Could not fit" in str(w.message)]
        assert len(fit_warnings) == 1
        assert "365 of 365" in str(fit_warnings[0].message)

    def test_user_freq_col_named_freq_is_preserved(self, monthly_df):
        df = monthly_df.copy()
        df["freq"] = pd.to_datetime(df["date"]).dt.month
        out = SPI().calculate(df, "date", "TotalPrecipitation", freq_col="freq")
        assert "freq" in out.columns
        by_freq = SPI().calculate(monthly_df, "date", "TotalPrecipitation", freq="M")
        np.testing.assert_allclose(out[INDEX_COL].values, by_freq[INDEX_COL].values)


class TestReturnParams:
    def test_shape_and_columns(self, monthly_df):
        df_spi, df_params = SPI().calculate(
            monthly_df, "date", "TotalPrecipitation", freq="M", return_params=True
        )
        assert len(df_params) == 12
        for col in ["column", "freq_group", "dist_type", "fit_type", "n_fit", "p_zero"]:
            assert col in df_params.columns
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
            monthly_df,
            "date",
            "TotalPrecipitation",
            freq="M",
            dist_type="nor",
            return_params=True,
        )
        assert df_params["p_zero"].isna().all()
