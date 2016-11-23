import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import os
from plot_index import plot_index
from spi import SPI

def create_datelist(start_date, n_months):
    
    dates = [start_date + relativedelta(months=i) 
              for i in range(0, n_months)]
    
    return np.array(dates)


if __name__=='__main__':
    # Read precip data from csv
    crnt_path = os.path.dirname(os.path.abspath(__file__))
    precip_file = os.path.join(crnt_path,'..','data','rainfall_test.csv')
    rainfall_data = np.genfromtxt(precip_file, delimiter=',')
    
    # Initialize SPI class
    test_spi = SPI() 
    
    # Set rolling window parameters
    test_spi.set_rolling_window_params(
        span=1, window_type=None, center=True
    )
    
    # Set distribution parameters
    test_spi.set_distribution_params(dist_type='gamma')
    
    # Calculate SPI
    data = test_spi.calculate(rainfall_data, starting_month=1)
    
    # Create date list for plotting
    n_dates = np.shape(data)[0]
    date_list = create_datelist(dt.date(2000,1,1), n_dates)
    
    # Plot
    #plot_index(date_list, data)
    print np.squeeze(data)

    print len(data)