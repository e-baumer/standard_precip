# Standard Precipitation (Evapotranspiration) Index

[![CI](https://github.com/e-baumer/standard_precip/actions/workflows/ci.yml/badge.svg)](https://github.com/e-baumer/standard_precip/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/standard-precip)](https://pypi.org/project/standard-precip/)
[![GitHub license](https://img.shields.io/github/license/e-baumer/standard_precip)](https://github.com/e-baumer/standard_precip/blob/master/LICENSE)

## Overview

This is a Python implementation for calculating the Standardized Precipitation Index (SPI) and the
Standardized Precipitation Evapotranspiration Index (SPEI) — key indices for identifying droughts.
See [NCAR's Climate Data Guide](https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-evapotranspiration-index-spei)
for a useful discussion of the relative merits of SPI vs SPEI.

These functions are loosely based on the [SPEI package in R](https://cran.r-project.org/web/packages/SPEI/SPEI.pdf)
by Santiago Beguería and Sergio M. Vicente-Serrano. The paper that most clearly describes the
algorithm is:

> Lloyd-Hughes, Benjamin, and Mark A. Saunders. "A drought climatology for Europe."
> International Journal of Climatology 22.13 (2002): 1571-1592.

This paper is included in the docs folder.

Distributions can be fit with either L-moments or Maximum Likelihood Estimation (MLE).
**L-moments is the recommended method** — it is what NCAR's SPI calculators and R's SPI/SPEI
packages use, and MLE can be unstable for some distributions and datasets. Indices can be
calculated on daily, weekly, monthly, annual or any custom time grouping, at any scale
(e.g. 3-month, 6-month), and optionally against a fixed baseline (reference) period.

Requires Python 3.10+.

## Installation

```
pip install standard-precip
```

or with [uv](https://docs.astral.sh/uv/):

```
uv add standard-precip
```

## Available distributions

Distribution | L-moments | MLE
:----------- | :-------- | :--
Gamma | :heavy_check_mark: | :heavy_check_mark:
Exponential | :heavy_check_mark: | :heavy_check_mark:
Generalized Extreme Value | :heavy_check_mark: | :heavy_check_mark:
Generalized Pareto | :heavy_check_mark: | :heavy_check_mark:
Gumbel | :heavy_check_mark: | :heavy_check_mark:
Normal | :heavy_check_mark: | :heavy_check_mark:
Pearson III | :heavy_check_mark: | :heavy_check_mark:
Weibull | :heavy_check_mark: | :heavy_check_mark:
Generalized Logistic | :heavy_check_mark: | :heavy_check_mark:
Generalized Normal | :heavy_check_mark: | :heavy_check_mark:
Kappa | :heavy_check_mark:¹ | :heavy_check_mark:
Wakeby | :heavy_check_mark: | -

¹ Kappa L-moment ratios have no solution for some samples; affected groups yield NaN with a warning.

Note: for the Generalized Logistic, Generalized Normal and Kappa distributions, the L-moments
fits use Hosking's distributions (via [lmoments3](https://github.com/Ouranosinc/lmoments3)) while
the MLE fits use scipy's `genlogistic`, `gennorm` and `kappa4`. These are different
parameterizations — parameters from one fit type are not interchangeable with the other.

## Basic usage

For a more detailed walkthrough see the [example notebook](examples/example_use.ipynb).

```python
import pandas as pd
from standard_precip import SPI
from standard_precip.utils import plot_index

rainfall_data = pd.read_csv('data/monthly_data.csv')

spi = SPI()
df_spi = spi.calculate(
    rainfall_data,
    'date',
    'TotalPrecipitation',
    freq="M",
    scale=1,
    fit_type="lmom",
    dist_type="gam"
)
```

For a 3-month SPI, set `scale=3` (the precipitation is summed over a 3-month rolling window and
the result appears in a `..._scale_3_calculated_index` column). `precip_cols` may also be a list
of column names to process several series at once.

Plotting:

```python
fig = plot_index(df_spi, 'date', 'TotalPrecipitation_calculated_index')
```

### Baseline (reference) period

For climate-projection work you typically fit the distributions on a historical baseline and
apply them to the full record:

```python
df_spi = spi.calculate(
    rainfall_data, 'date', 'TotalPrecipitation',
    baseline_start=1961, baseline_end=1990   # ints = years; dates work too
)
```

### Fitted parameters

`return_params=True` additionally returns a dataframe of the fitted distribution parameters,
one row per (column, frequency group), including the probability of zero precipitation:

```python
df_spi, df_params = spi.calculate(
    rainfall_data, 'date', 'TotalPrecipitation', return_params=True
)
```

### Annual and custom frequencies

The `freq` parameter controls the seasonal grouping used for fitting: each day of year
(`freq="D"`), week of year (`"W"`) or month of year (`"M"`) forms its own fitting population.
For annual totals there is no seasonal cycle to condition on — use `freq=None` so that a single
distribution is fit to the whole series:

```python
df_annual = df.resample('YE', on='date').sum().reset_index()
df_spi = spi.calculate(df_annual, 'date', 'TotalPrecipitation', freq=None)
```

(Do not create a per-year grouping column for annual data: each year would end up alone in its
own fitting group.)

For any other custom grouping, add an integer column that assigns each row to a fitting group
and pass its name as `freq_col`.

### SPEI

SPEI is calculated exactly like SPI but on the climatic water balance D = P − PET. This package
does not compute potential evapotranspiration; supply the water-balance column yourself. Because
D takes negative values, use a distribution defined on the whole real line — Vicente-Serrano
et al. (2010) recommend the generalized logistic distribution:

```python
from standard_precip import SPEI

df['wb'] = df['precip'] - df['pet']
df_spei = SPEI().calculate(df, 'date', 'wb', dist_type='glo', fit_type='lmom')
```

## Notes

1. Although the user is allowed to select the distribution they wish to fit, one should be aware
   of the support of each distribution. Precipitation data can have zero values (handled for
   Gamma and Pearson III via the mixed CDF of Thom, 1966) and P − PET can take on negative
   values. This should be considered when selecting a distribution.
2. For gamma MLE fits, the location parameter is fixed at 0 unless you pass `floc`/`loc`
   explicitly; unconstrained-loc gamma MLE is ill-posed on precipitation data and produced
   extreme index values.
3. L-moment fitting is provided by the [lmoments3](https://github.com/Ouranosinc/lmoments3)
   package, which is licensed under GPL-3. This package's own code remains Apache-2.0;
   lmoments3 is an external dependency installed alongside it.

## Development

```
uv sync --group dev
uv run pytest
uv run ruff check .
```

## Contacts

Author - Eric Nussbaumer ([ebaumer@gmail.com](mailto:ebaumer@gmail.com))

## License

    Apache License, Version 2.0

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
