import warnings
from functools import reduce

import numpy as np
import pandas as pd
import scipy.stats as scs

from standard_precip import _distributions


class BaseStandardIndex():
    '''
    Calculate the SPI or SPEI index. A user specified distribution is fit to the precip data.
    The CDF of this distribution is then calculated after which the the standard normal
    distribution is calculated which gives the index. A distribution can be fit over the
    precipitation data either using MLE or L-moments. NCAR's SPI calculators and the SPI and
    SPEI R packages both use L-moments to fit the distribution. There are advantages and
    disadvantages to each technique.

    This calculation can be done on any time scale. Built in temporal scales include daily,
    weekly, and monthly; however, the user can define their own timescale.

    One should put some thought into the type of distribution fit to the
    data. Precipitation can have zero value and some distributions are only
    defined over interval (0, inf). Python's gamma distribution is defined
    over [0, inf). In addition SPEI which is constructed from precipitation
    - PET or (P-PET) can take on negative values.
    '''

    #: Distributions that are undefined at zero; zero observations are removed
    #: before fitting and handled through the mixed CDF (Thom, 1966).
    non_zero_distr = ['gam', 'pe3']

    @staticmethod
    def rolling_window_sum(df: pd.DataFrame, precip_cols: list, span: int = 1,
                           window_type: str | None = None, center: bool = False, **kwargs):
        '''
        This is a helper method which will find the rolling sum of precipitation data.
        Returns a new DataFrame; the input DataFrame is not modified.
        '''
        df = df.copy()
        precip_cols_new = []
        for p in precip_cols:
            new_col_name = p + f"_scale_{span}"
            df[new_col_name] = df[p].rolling(
                window=span, win_type=window_type, center=center, **kwargs
            ).sum()
            precip_cols_new.append(new_col_name)

        return df, precip_cols_new

    @staticmethod
    def check_duplicate_dates(df, date_col):
        '''
        Method to check duplicate dates in dataframe. If duplicates are found, the row corresponding
        to the first date found is used.
        '''
        if df.duplicated(subset=date_col).any():
            warnings.warn(
                "Found duplicate dates in dataframe. Removing duplicates and using "
                "first date found",
                UserWarning,
                stacklevel=3,
            )
            df = df.drop_duplicates(subset=date_col)

        return df

    def fit_distribution(self, data: np.ndarray, dist_type: str, fit_type: str = 'lmom',
                         **kwargs):
        '''
        Fit given distribution to historical precipitation data.
        The fit is accomplished using either L-moments or MLE (Maximum Likelihood Estimation).

        For distributions that use the Gamma Function (Gamma and Pearson 3) remove observations
        that have 0 precipitation values and fit using non-zero observations. Also find probability
        of zero observation (estimated by number of zero obs / total obs). This is for latter use
        in calculating the CDF using (Thom, 1966. Some Methods of Climatological Analysis)

        Returns a tuple of (distribution, params, p_zero) where params is None when there is
        not enough data to fit the distribution.
        '''

        # Get distribution type
        spec = _distributions.get_spec(dist_type)
        if fit_type == 'lmom':
            distrb = spec.lmom_dist
        elif fit_type == 'mle':
            distrb = spec.mle_dist
        else:
            raise ValueError(f"{fit_type} is not an option. Option fit_types are mle and lmom")
        if distrb is None:
            supported = 'L-moments' if spec.lmom_dist is not None else 'MLE'
            raise ValueError(f"'{dist_type}' supports {supported} fitting only")

        # Determine zeros if distribution can not handle x = 0
        p_zero = None
        if dist_type in self.non_zero_distr:
            p_zero = data[data == 0].shape[0] / data.shape[0]
            data = data[data != 0]

        min_samples = _distributions.min_samples(spec, fit_type)

        if (data.shape[0] < min_samples) or (p_zero is not None and np.isclose(p_zero, 1.0)):
            warnings.warn(
                f"Insufficient data to fit '{dist_type}' distribution "
                f"({data.shape[0]} non-zero observations, {min_samples} required); "
                "returning NaN for this group.",
                UserWarning,
                stacklevel=3,
            )
            params = None

        else:
            try:
                distrb, params = _distributions.fit(spec, data, fit_type, **kwargs)
            except ValueError as err:
                # e.g. kappa L-moment ratios can be unsolvable for some samples
                warnings.warn(
                    f"Could not fit '{dist_type}' distribution ({err}); "
                    "returning NaN for this group.",
                    UserWarning,
                    stacklevel=3,
                )
                params = None

        return distrb, params, p_zero

    def cdf_to_ppf(self, data, distrb, params, p_zero):
        '''
        Take the specific distributions fitted parameters and calculate the
        cdf. Apply the inverse normal distribution to the cdf to get the SPI
        SPEI. This process is best described in Lloyd-Hughes and Saunders, 2002
        which is included in the documentation.
        '''

        # Calculate the CDF of observed precipitation on a given time scale
        if params:
            if p_zero is not None:
                cdf = p_zero + (1 - p_zero) * distrb.cdf(data, **params)
            else:
                cdf = distrb.cdf(data, **params)
        else:
            cdf = np.full(np.shape(data), np.nan)

        # Apply inverse normal distribution
        norm_ppf = scs.norm.ppf(cdf)
        norm_ppf[np.isinf(norm_ppf)] = np.nan

        return norm_ppf

    @staticmethod
    def _baseline_timestamp(value, end: bool):
        '''Normalize a baseline bound to a pd.Timestamp. Integers are treated as
        years: the start of the year for the lower bound, the end of the year
        for the upper bound (both inclusive).'''
        if isinstance(value, int):
            if end:
                return pd.Timestamp(year=value, month=12, day=31, hour=23, minute=59, second=59)
            return pd.Timestamp(year=value, month=1, day=1)
        return pd.Timestamp(value)

    def calculate(self, df: pd.DataFrame, date_col: str, precip_cols: list,
                  freq: str | None = "M", scale: int = 1, freq_col: str | None = None,
                  fit_type: str = 'lmom', dist_type: str = 'gam', baseline_start=None,
                  baseline_end=None, return_params: bool = False,
                  **dist_kwargs) -> pd.DataFrame:
        '''
        Calculate the index.

        Check https://docs.scipy.org/doc/scipy/reference/stats.html for
        distribution types

        Parameters
        ----------
        df: pd.Dataframe
            Pandas dataframe with precipitation data as columns. Each column is treated as a
            separate set of observations and distributions are fit for individual columns. A date
            column should also be given in the dataframe.

        date_col: str
            The column name for the date column. Date specification should follow the strftime format.

        precip_cols: list
            List of columns with precipitation data. Each column is treated as a separate set of
            observations.

        freq: str or None ["M", "W", "D", None]
            The temporal frequency to calculate the index on. The day of year ("D") or week of year
            ("W") or month of year ("M") is derived from the date_col. Use freq=None when the data
            has no seasonal cycle to condition on - most commonly annual totals - so that a single
            distribution is fit to the entire column. Do NOT pass a per-year freq_col for annual
            data: every year would end up alone in its own fitting group. If the user desires a
            custom frequency such as 3-month, 6-month, they can pass the column name for the custom
            frequency (freq_col)

        freq_col: str (column type: int)
            Name of the column that specifies a custom frequency. This overrides the freq parameter.
            The freq_col should group individual observations (rows) according to the users custom
            frequency. The grouping is specified using integers.

        scale: int (default=1)
            Integer to specify the number of time periods over which the standardized precipitation
            index is to be calculated. If freq="M" then this is the number of months.

        fit_type: str ("lmom" or "mle")
            Specify the type of fit to use for fitting distribution to the precipitation data. Either
            L-moments (lmom) or Maximum Likelihood Estimation (mle). Note use L-moments when comparing
            to NCAR's NCL code and R's packages to calculate SPI and SPEI.

        dist_type: str
            The distribution type to fit using either L-moments or MLE
                'gam' - Gamma
                'exp' - Exponential
                'gev' - Generalised Extreme Value
                'gpa' - Generalised Pareto
                'gum' - Gumbel
                'nor' - Normal
                'pe3' - Pearson III
                'wei' - Weibull

            The distribution type to fit using ONLY MLE
                'glo' - Generalised Logistic
                'gno' - Generalised Normal
                'kap' - Kappa

            The distribution type to fit using ONLY L-moments
                'wak' - Wakeby

        baseline_start, baseline_end: int, str or pd.Timestamp, optional
            Reference (baseline) period over which the distributions are fit; the fitted
            distributions (and the probability of zero precipitation) are then used to
            transform the entire record. Integers are interpreted as years, inclusive on
            both ends (e.g. baseline_start=1961, baseline_end=1990); strings and Timestamps
            are interpreted as dates. Both bounds must be given together. This is the usual
            setup for climate-projection work: fit on a historical baseline, apply to the
            projected record. By default the full record is used for fitting.

        return_params: bool (default=False)
            If True, additionally return a dataframe of the fitted distribution parameters,
            with one row per (precipitation column, frequency group): the number of
            observations used in the fit, the probability of zero precipitation (for
            distributions where zeros are removed before fitting) and the fitted parameters.

        dist_kwargs:
            scale and location parameters. See documentation on scipy.stats.rv_continuous.fit

        Returns
        -------
        df: pd.Dataframe
            Pandas dataframe with the calculated indices for each precipitation column appended
            to the original dataframe. If return_params is True, a tuple of
            (df, params_dataframe) is returned instead.
        '''

        # Check for duplicate dates
        df = self.check_duplicate_dates(df, date_col)
        if isinstance(precip_cols, str):
            precip_cols = [precip_cols]

        if scale > 1:
            df, precip_cols = self.rolling_window_sum(df, precip_cols, scale)

        if (baseline_start is None) != (baseline_end is None):
            raise ValueError("baseline_start and baseline_end must be given together")

        keep_cols = [date_col] + precip_cols
        if freq_col is not None:
            if freq_col not in df.columns:
                raise ValueError(f"freq_col '{freq_col}' is not a column of the dataframe")
            keep_cols.append(freq_col)
        df_copy = df[keep_cols].copy()
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

        if freq_col is None:
            freq_col = 'freq'

            if freq == "D":
                df_copy[freq_col] = df_copy[date_col].dt.dayofyear
            elif freq == "W":
                df_copy[freq_col] = df_copy[date_col].dt.isocalendar().week.astype(int)
            elif freq == "M":
                df_copy[freq_col] = df_copy[date_col].dt.month
            elif freq is None:
                # No seasonal conditioning (e.g. annual totals): the whole series
                # forms a single fitting population.
                df_copy[freq_col] = 0
            else:
                raise ValueError(
                    f"{freq} is not a recognized frequency. Options are 'M', 'W', 'D' or None"
                )
        elif not pd.api.types.is_integer_dtype(df_copy[freq_col]):
            raise ValueError(
                f"freq_col '{freq_col}' must be an integer column grouping the observations"
            )

        baseline_mask = None
        if baseline_start is not None:
            start_ts = self._baseline_timestamp(baseline_start, end=False)
            end_ts = self._baseline_timestamp(baseline_end, end=True)
            if start_ts > end_ts:
                raise ValueError(
                    f"baseline_start ({start_ts}) is after baseline_end ({end_ts})"
                )
            baseline_mask = df_copy[date_col].between(start_ts, end_ts)
            if not baseline_mask.any():
                raise ValueError(
                    f"No observations fall within the baseline period {start_ts} - {end_ts}"
                )

        freq_range = df_copy[freq_col].unique().tolist()
        # Loop over the frequency groups (e.g. months of the year)
        dfs = []
        params_rows = []
        for p in precip_cols:
            dfs_p = []
            for j in freq_range:
                precip_all = df_copy.loc[df_copy[freq_col] == j]
                precip_single_df = precip_all.dropna().copy()
                precip_single = precip_single_df[p].values

                # Fit distribution for particular series and frequency group. With a
                # baseline period, the distribution (and p_zero) is fit on the baseline
                # observations only, then applied to the whole group.
                if baseline_mask is not None:
                    fit_values = precip_single_df.loc[
                        baseline_mask.loc[precip_single_df.index], p
                    ].values
                else:
                    fit_values = precip_single
                distrb, params, p_zero = self.fit_distribution(
                    fit_values, dist_type, fit_type, **dist_kwargs
                )

                if return_params:
                    row = {
                        'column': p,
                        'freq_group': j,
                        'dist_type': dist_type,
                        'fit_type': fit_type,
                        'n_fit': fit_values.shape[0],
                        'p_zero': p_zero,
                    }
                    if params:
                        row.update(params)
                    params_rows.append(row)

                # Calculate SPI/SPEI
                spi = self.cdf_to_ppf(precip_single, distrb, params, p_zero)
                idx_col = f"{p}_calculated_index"
                precip_single_df[idx_col] = spi
                dfs_p.append(precip_single_df[[date_col, idx_col]])
            dfs.append(pd.concat(dfs_p).sort_values(date_col))

        df_all = reduce(
            lambda left, right: pd.merge(left, right, on=date_col, how='left'), dfs, df_copy
        )
        if freq_col == 'freq':
            df_all = df_all.drop(columns=freq_col)

        if return_params:
            df_params = pd.DataFrame(params_rows).sort_values(
                ['column', 'freq_group']
            ).reset_index(drop=True)
            return df_all, df_params

        return df_all
