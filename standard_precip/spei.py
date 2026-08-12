from standard_precip.base_sp import BaseStandardIndex


class SPEI(BaseStandardIndex):
    '''
    Calculate the Standardized Precipitation Evapotranspiration Index (SPEI).

    SPEI is computed exactly like SPI, but on the climatic water balance
    D = P - PET (precipitation minus potential evapotranspiration) instead of
    precipitation alone. This package does not compute PET; supply a column
    containing the water balance and pass it as the precipitation column:

        spei = SPEI()
        df_spei = spei.calculate(df, 'date', 'water_balance',
                                 dist_type='glo', fit_type='lmom')

    Because D takes negative values, distributions restricted to positive
    support - gamma ('gam') and Pearson III ('pe3'), for which zero
    observations are stripped and handled through the mixed CDF - are not
    appropriate. Vicente-Serrano et al. (2010), which introduced SPEI,
    recommends the three-parameter log-logistic distribution: use
    dist_type='glo' with fit_type='lmom'.

    Reference: Vicente-Serrano, S.M., Begueria, S., Lopez-Moreno, J.I. (2010).
    A Multiscalar Drought Index Sensitive to Global Warming: The Standardized
    Precipitation Evapotranspiration Index. Journal of Climate, 23(7).
    '''
