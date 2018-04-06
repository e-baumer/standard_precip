import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import scipy.stats
from standard_precip.plot_index import plot_index
from standard_precip.spi import SPI


def create_datelist(start_date, n_months):

    dates = [start_date + relativedelta(months=i) for i in range(0, n_months)]

    return np.array(dates)


# Read precip data from csv
crnt_path = os.path.dirname(os.path.abspath(__file__))
precip_file = os.path.join(crnt_path, 'data', 'rainfall_test.csv')
rainfall_data = np.genfromtxt(precip_file, delimiter=',')

# Initialize SPI class
spi = SPI()

# Set rolling window parameters
spi.set_rolling_window_params(
    span=1, window_type=None, center=True
)

# Set distribution parameters
spi.set_distribution_params(dist_type='gam')

# Calculate SPI
data = spi.calculate(rainfall_data, starting_month=1)

# Create date list for plotting
n_dates = np.shape(data)[0]
date_list = create_datelist(dt.date(2000, 1, 1), n_dates)

# Plot SPI
plot_index(date_list, data)
print(np.squeeze(data))


# Test find best distribution fit
dist_list = ['gam', 'exp', 'wei']

test_data = scipy.stats.gamma.rvs(10., size=100)
sse = spi.best_fit_distribution(test_data, dist_list, bins=20, save_file='test.jpg')

for k in sse:
    print("distribution = {0:}; SSE = {1:}".format(k, sse[k]))
