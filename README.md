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


## Example Use

Imports
```
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import os
from plot_index import plot_index
from standard_precip.spi import SPI
```

A useful function for calculating a list of dates
```
def create_datelist(start_date, n_months):
    
    dates = [start_date + relativedelta(months=i) 
              for i in range(0, n_months)]
    
    return np.array(dates)
```

Read example monthly precipitation data (included in data folder).
```
rainfall_data = np.genfromtxt('rainfall_test.csv', delimiter=',')
```

For this example we will calculate SPI, therefore initialize the SPI class
```
spi = SPI()
```

Set rolling window average parameters. In this example since window_type is None
we don't actually implement a rolling window.
```
spi.set_rolling_window_params(
    span=1, window_type=None, center=True
)
```
Set statistical distribution fit parameters. When calling SPI class the default
distribution is a generalized gamma distribution which is a three parameter gamma
distribution. Here we set it to a gamma distribution (two parameters) for no reason.
```
spi.set_distribution_params(dist_type='gam')
```

Calculate SPI. The parameter starting_month indicates the month at which the 
data starts.
```
data = spi.calculate(rainfall_data, starting_month=1)
```
Create a date list for plotting.
```
n_dates = np.shape(data)[0]
date_list = create_datelist(dt.date(2000,1,1), n_dates)
```

Plot data
```
plot_index(date_list, data)
```

## TO DO
1. Implement calculations of PET
3. Finish generator to process large datasets
4. Add metric for fit of distribution to historical data

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
