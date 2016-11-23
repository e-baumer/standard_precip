# [Standard Precipitation (Evapotranspiration) Index](http://sac.csic.es/spei/home.html)
<!--[![Build Status]()]()-->


## Overview
This is a Python implementation for calculating the Standard Precipitation Index
(SPI) and the Standard Precipipation Evapotranspiration Index (SPEI). These are
two key indicies in identifying droughts. See [NCAR's Climate Data Guide]
(https://climatedataguide.ucar.edu/climate-data/standardized-precipitation-evapotranspiration-index-spei) for a usefull discussion 
of the relative merits of SPI vs SPEI. 

These functions are loosely based on the [SPEI package in R](https://cran.r-project.org/web/packages/SPEI/SPEI.pdf) by Santiago Beguería and Sergio M. Vicente-Serrano.

There are many papers on SPI and SPEI. I found the paper which most clearly 
describes the algorithms is:
	Lloyd‐Hughes, Benjamin, and Mark A. Saunders. "A drought climatology for Europe." International journal of climatology 22.13 (2002): 1571-1592.
This paper is included in the docs folder.

The underlying algorithm to calculate SPI and SPEI are the same; however, SPI is
calculated using precipitation data only, whereas SPEI is calculated using precipiation
minus potential evapotranspiration (PET).

There is some consensus in the literature as to which distribution to fit historical data. 
For precipitation data only (SPI) it is suggested to use a Gamma distribution. This 
is the default distribution in the SPI function. For SPEI, some have suggested using
a log-logistic distribution. The default distribution for the SPEI function is
the Fisk or log-logistic distribution. However, the user can select their own 
distribution (see Notes).

## Example Use


## TO DO
1. Implement calculations of PET
2. Add plotting
3. Finish generator to process large datasets

## Notes
1. Although the user is allowed to select the distribution (from scipy stats)
that they wish to fit historical data to, one should be aware of the support of 
each particular distribution. Precipitation data can have zero values and P-PEI 
can take on negative values. This should be considered when selecting a distribution.
2.

## Build


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