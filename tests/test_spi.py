from standard_precip import spi
import numpy as np


def test_gamma_out():

    SPI = spi.SPI()
    rainfall_data = np.genfromtxt('data/rainfall_test2.csv', delimiter=',')
    SPI.set_rolling_window_params(span=10, window_type='boxcar', center=False)
    SPI.set_distribution_params(dist_type='gam')
    result = SPI.calculate(rainfall_data, starting_month=1)
    assert np.round(result[-1][0], 4) == np.round(-0.09562831, 4)
