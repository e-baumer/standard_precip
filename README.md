# [Standard Precipitation (Evapotranspiration) Index](http://sac.csic.es/spei/home.html)
[![Build Status](https://travis-ci.org/e-baumer/standard_precip.svg?branch=master)](https://travis-ci.org/e-baumer/standard_precip)
[![GitHub license](https://img.shields.io/github/license/e-baumer/standard_precip)](https://github.com/e-baumer/standard_precip/blob/master/LICENSE)

## Overview
This is a Python implementation for calculating the Standard Precipitation Index
(SPI). This is one of the key indicies in identifying droughts. See [NCAR's Climate Data Guide]
(https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-evapotranspiration-index-spei) for a usefull discussion 
of the relative merits of SPI vs SPEI. 

These functions are loosely based on the [SPEI package in R](https://cran.r-project.org/web/packages/SPEI/SPEI.pdf) by Santiago Beguería and Sergio M. Vicente-Serrano.

There are many papers on SPI and SPEI. I found the paper which most clearly 
describes the algorithms is:
	Lloyd‐Hughes, Benjamin, and Mark A. Saunders. "A drought climatology for Europe." International journal of climatology 22.13 (2002): 1571-1592.
This paper is included in the docs folder.

There is some consensus in the literature as to which distribution to fit historical data. 
For precipitation data only (SPI) it is suggested to use a Gamma distribution. This 
is the default distribution in the SPI function.  However, the user can select their own 
distribution (see Notes).

The current implementation allows for the user to fit precipitation data with using either L-moments or Maximum
Likelihood Estimation (MLE). It also allows for the fitting of daily, weekly, monthly or any custom time frame
of SPI data.

Currently on compatible with Python3.

## Available Distributions
The following is a table of distributions used to fit the precipitation data. The table indicates whether the
distribution is available for L-moments or MLE.

Distribution | L-Moments | MLE
:----------- | :---------- | :--------
Gamma |:heavy_check_mark: | :heavy_check_mark:
Exponential |:heavy_check_mark: | :heavy_check_mark:
Generalized Extreme Value |:heavy_check_mark: | :heavy_check_mark:
Generalized Pareto |:heavy_check_mark: | :heavy_check_mark:
Gumbel |:heavy_check_mark: | :heavy_check_mark:
Normal |:heavy_check_mark: | :heavy_check_mark:
Pearson III |:heavy_check_mark: | :heavy_check_mark:
Weibull |:heavy_check_mark: | :heavy_check_mark:
Generalized Logistic | - | :heavy_check_mark:
Generalized Normal | - | :heavy_check_mark:
Wakeby | :heavy_check_mark: | -

## Installation
```
pip install standard-precip
```


## Basic Usage 

For more detailed example see the example notebook.

Imports
```
import pandas as pd
from standard_precip.spi import SPI
from standard_precip.utils import plot_index
```

The SPI function expects the data to be in a Pandas DataFrame
Read example monthly precipitation data (included in data folder).
```
rainfall_data = pd.read_csv('monthly_data.csv')
```

For this example we will calculate SPI, therefore initialize the SPI class
```
spi = SPI()
```

Calculate the 1-Month SPI using Gamma function and L-moments. You must indicate the date column and the 
precipitation column of the DataFrame. You can have a list of precipitation columns to process.
```
df_spi = new_spi.calculate(
    rainfall_data, 
    'date', 
    'precip', 
    freq="M", 
    scale=1, 
    fit_type="lmom", 
    dist_type="gam"
)
```

Calculate the 3-Month SPI using Gamma function and L-moments. You must indicate the date column and the 
precipitation column of the DataFrame. You can have a list of precipitation columns to process.
```
df_spi = new_spi.calculate(
    rainfall_data, 
    'date', 
    'precip', 
    freq="M", 
    scale=3, 
    fit_type="lmom", 
    dist_type="gam"
)
```
The freq parameter indicates the type of data you are using, daily, weekly, monthly. However, if you have a custom
time period you are interested you can over-ride the freq parameter by using creating a column in the DataFrame for
grouping the observations and indicating this column in the freq_col parameter. The distributions and indicies will
be calculated using the integer grouping in the freq_col.

Plot data
```
fig = plot_index(df_spi, 'date', 'precip_scale_3_calculated_index')
```

## TO DO
1. Implement calculations of PET for SPEI
2. Add other drought indicators
3. Create functionality for finding best distribution based on data

## Notes
1. Although the user is allowed to select the distribution (from scipy stats)
that they wish to fit historical data to, one should be aware of the support of 
each particular distribution. Precipitation data can have zero values and P-PEI 
can take on negative values. This should be considered when selecting a distribution.


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
