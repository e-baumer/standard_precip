import scipy
import numpy as np
import pandas as pd
import datetime as dt
import scipy.stats as scs
from functools import reduce
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta

from .lmoments import distr


class BaseStandardIndex(object):
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

    def __init__(self):
        self.distrb = None
        self.non_zero_distr = ['gam', 'pe3']

    @staticmethod
    def rolling_window_sum(df: pd.DataFrame, span: int, window_type, center, **kwargs):
        '''
        This is a helper method which will find the rolling sum of precipitation data.
        '''
        window_sum = np.squeeze(data_df.rolling(
            window=span, win_type=window_type, center=center, **kwargs
        ).sum().values.T)

        return window_sum

    @staticmethod
    def rolling_window_mean(self, data, span, window_type, center, **kwargs):
        '''
        This is a helper method which will find the rolling mean of precipitation data.
        '''
        data_df = pd.DataFrame(data)

        window_mean = np.squeeze(data_df.rolling(
            window=span, win_type=window_type, min_periods=span,
            center=center, **kwargs
        ).mean().values.T)

        return window_mean

    @staticmethod
    def check_duplicate_dates(df, date_col):
        '''
        Method to check duplicate dates in dataframe. If duplicates are found, the row corresponding
        to the first date found is used.
        '''
        if df.duplicated(subset=date_col).any():
            print("Found duplicate dates in dataframe. Removing duplicates and using first date found")
            df = df.drop_duplicates(subset=date_col)

        return df

    @staticmethod
    def best_fit_distribution(data: np.array, dist_list: list, fit_type: str='lmom',
                              bins: int=10, save_file: str=None):
        '''
        Method to find the best distribution for observational data. Calculates the Sum of the
        Squares error between fitted distribution and pdf.
        Inspired by: http://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

        Parameters
        ----------
        data: np.array size: [Number Observations, ]
            A numpy array of size [Number Observations, ] with the precipiation data.

        dist_type: list
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


        fit_type: str ("lmom" or "mle")
            Specify the type of fit to use for fitting distribution to the precipitation data. Either
            L-moments (lmom) or Maximum Likelihood Estimation (mle). Note use L-moments when comparing
            to NCAR's NCL code and R's packages to calculate SPI and SPEI.

        bins: int
            Number of bins to bin precipitation data

        save_file: str
            File path and name to save figure of precipitation data and fitted distributions.

        Returns
        -------
        sse: dict (key - distribution, value - sum of square error)
            The sum of the squares error between fitted distribution and pdf.
        '''
        y, x = np.histogram(data, bins=bins, normed=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0

        sse = {}
        fig, ax = plt.subplots()
        ax.bar(x, y, width=0.5, align='center', color='b', alpha=0.5, label='data')

        for i, dist_name in enumerate(dist_list):
            distrb = getattr(distr, dist_type)

            if fit_type == 'lmom':
                params = distrb.lmom_fit(data, **kwargs)

            elif fit_type == 'mle':
                params = distrb.fit(data, **kwargs)

            else:
                raise AttributeError(f"{fit_type} is not an option. Option fit_types are mle and lmom")

            pdf = dist.pdf(x, **params)
            sse[dist_name] = np.sum((y - pdf)**2)
            ax.plot(x, pdf, label=dist_name)

        ax.legend()
        ax.grid(True)

        if save_file:
            plt.savefig(save_file, dpi=400)
        else:
            plt.show()

        sse = sorted(sse.items(), key=lambda x: x[1], reverse=False)
        return sse

    def fit_distribution(self, data: np.array, dist_type: str, fit_type: str='lmom', **kwargs):
        '''
        Fit given distribution to historical precipitation data.
        The fit is accomplished using either L-moments or MLE (Maximum Likelihood Estimation).

        For distributions that use the Gamma Function (Gamma and Pearson 3)
        '''

        # Get distribution type
        self.distrb = getattr(distr, dist_type)

        # Determine zeros if distribution can not handle x = 0
        p_zero = None
        if dist_type in self.non_zero_distr:
            p_zero = data[data == 0].shape[0] / data.shape[0]
            data = data[data != 0]

        # Fit distribution
        if fit_type == 'lmom':
            params = self.distrb.lmom_fit(data, **kwargs)

        elif fit_type == 'mle':
            params = self.distrb.fit(data, **kwargs)

        else:
            raise AttributeError(f"{fit_type} is not an option. Option fit_types are mle and lmom")

        return params, p_zero

    def calculate(self, df: pd.DataFrame, date_col: str, precip_cols: list, freq: str="M",
                  freq_col: str=None, fit_type: str='lmom', dist_type: str='gam', **dist_kwargs):
        '''
        Calculate the index.

        Check https://docs.scipy.org/doc/scipy/reference/stats.html for
        distribution types

        Parameters
        ----------
        df: pd.Dataframe
            Pandas dataframe with precipitation data as columns. Each column is treated as a seperate
            set of observations and distributions are fit for individual columns. A date column should
            also be given in the dataframe.

        date_col: str
            The column name for the date column. Date specification should follow the strftime format.

        precip_cols: list
            List of columns with precipitation data. Each column is treated as a separate set of
            observations.

        freq: str ["M", "W", "D"]
            The temporal frequency to calculate the index on. The day of year ("D") or week of year
            ("W") or month of year ("M") is derived from the date_col. If the user desires a custome
            frequency such as 3-month, 6-month, they can pass the column name for the custome freqency
            (freq_col)

        freq_col: str (column type: int)
            Name of the column that specifies a custome frequency. This overrides the freq parameter.
            The freq_col should group individual observations (rows) according to the users custome
            frequency. The grouping is specified using integers.

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

        dist_kwargs:
            scale and location parameters. See documentation on scipy.stats.rv_continuous.fit

        Returns
        -------
        df: pd.Dataframe
            Pandas dataframe with the calculated indicies for each precipitation column appended
            to the original dataframe.
        '''

        # Check for duplicate dates
        df = self.check_duplicate_dates(df, date_col)
        if isinstance(precip_cols, str):
            precip_cols = [precip_cols]

        # for p_col in precip_cols:
            # df.loc[df[p_col] == 0.0, p_col] = 0.001

        self._df_copy = df[[date_col] + precip_cols]
        self._df_copy[date_col] = pd.to_datetime(self._df_copy[date_col])

        if freq_col:
            self.freq_col = freq_col
        else:
            self.freq_col = 'freq'

            if freq == "D":
                self._df_copy[self.freq_col] = self._df_copy[date_col].dt.dayofyear
            elif freq == "W":
                self._df_copy[self.freq_col] = self._df_copy[date_col].dt.week
            elif freq == "M":
                self._df_copy[self.freq_col] = self._df_copy[date_col].dt.month
            else:
                raise AttributeError(f"{freq} is not a recognized frequency. Options are 'M', 'W', or 'D'")

        freq_range = self._df_copy[self.freq_col].unique().tolist()
        # Loop over months
        dfs = []
        for p in precip_cols:
            dfs_p = pd.DataFrame()
            for j in freq_range:
                precip_all = self._df_copy.loc[self._df_copy[self.freq_col]==j]
                precip_single_df = precip_all.dropna()
                precip_single = precip_single_df[p].values
                precip_sorted = np.sort(precip_single)[::-1]

                # Fit distribution for particular series and month
                params, p_zero = self.fit_distribution(
                    precip_sorted, dist_type, fit_type, **dist_kwargs
                )

                # Calculate SPI/SPEI
                spi = self.cdf_to_ppf(precip_single, params, p_zero)
                idx_col = f"{p}_calculated_index"
                precip_single_df[idx_col] = spi
                precip_single_df = precip_single_df[[date_col, idx_col]]
                dfs_p = pd.concat([dfs_p, precip_single_df])
                dfs_p = dfs_p.sort_values(date_col)
            dfs.append(dfs_p)

        df_all = reduce(
            lambda left, right: pd.merge(left, right, on=date_col, how='left'), dfs, self._df_copy
        )
        df_all = df_all.drop(columns=self.freq_col)

        return df_all

    def cdf_to_ppf(self, data, params, p_zero):
        '''
        Take the specific distributions fitted parameters and calculate the
        cdf. Apply the inverse normal distribution to the cdf to get the SPI
        SPEI. This process is best described in Lloyd-Hughes and Saunders, 2002
        which is included in the documentation.

        '''

        # Calculate the CDF of observed precipitation on a given time scale
        if p_zero:
            cdf = p_zero + (1 - p_zero) * self.distrb.cdf(data, **params)
        else:
            cdf = self.distrb.cdf(data, **params)

        # Apply inverse normal distribution
        norm_ppf = scs.norm.ppf(cdf)

        return norm_ppf
