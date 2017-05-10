from __future__ import absolute_import, division, print_function, unicode_literals
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import scipy
import scipy.stats
import types


def create_month_cycle(n_months, start_month=1):
    '''
    Create a repeating array of months 1-12, truncated by start_month
    Example:
             Create an array of 24 months starting on March (3)
             create_month_cycle(24, start_month=3)
             array([ 3,  4,  5,  6,  7,  8,  9, 10, 11, 12,  1,  2,  3,  4,  5,  
                     6,  7, 8,  9, 10, 11, 12,  1,  2])
    
    '''
    
    start = start_month - 1
    end = n_months + start_month - 1
    
    months = [(dt.datetime(2000,1,1) + relativedelta(months=i)).month 
              for i in range(start, end)]
    
    return np.array(months)


class BaseStandardIndex(object):
    '''
    '''
    
    def __init__(self):
        self.span = None
        self.window_type = None
        self.params = None
        self.dist_type = None
        self.rw_kwargs = None
        self.dist_kwargs = None
        self.rw_center = None
        self.distr = None
        
    
    def set_rolling_window_params(self, span=1, window_type=None, center=True, 
                                  **kwargs):
        '''
        span -- Size of the moving window. This is the number of observations 
                used for calculating the statistic. Each window will be a fixed 
                size.
        window_type -- includes 'boxcar', 'traing', 'blackman', etc
        rw_center -- If true calculated value is set to center of window, if 
                     false it is the right edge of window
        kwargs -- Additional arguments. See pandas rolling documentation.
        '''
        
        self.span = span
        self.window_type = window_type
        self.rw_center = center
        self.rw_kwargs = kwargs
    
    
    def set_distribution_params(self, dist_type='norm', **kwargs):
        '''
        dist_type -- Distribution type to be fit (Gamma, fisk, etc)
        kwargs -- Addition arguments for fit and calculation of cdf and ppf. 
                  These can include loc and scale. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous
        '''
        
        self.dist_type = dist_type
        self.dist_kwargs = kwargs
        
        
    def rolling_window_sum(self, data, span, window_type, center, **kwargs):
        '''
        
        pd.rolling_window is depreciated in pandas version 0.19
        '''

        data_df = pd.DataFrame(data)
        
        window_sum = np.squeeze(data_df.rolling(
            window=span, win_type=window_type, center=center, **kwargs
        ).sum().values.T)

        #weighted_sum = pd.rolling_window(
            #data, window=span, win_type=window_type, center=True, 
            #mean=False, **kwargs
        #)
        
        return window_sum
    
        
    def rolling_window_mean(self, data, span, window_type, center, **kwargs):
        '''
        pd.rolling_window is depreciated in pandas version 0.19. Fuck 2016
        '''
        
        data_df = pd.DataFrame(data)
        
        window_mean = np.squeeze(data_df.rolling(
            window=span, win_type=window_type, min_periods=span,
            center=center, **kwargs
        ).mean().values.T)
        
        #weighted_mean = pd.rolling_window(
            #data, window=span, win_type=window_type, center=True, 
            #mean=True, **kwargs
        #)

        return window_mean
    
    
    def fit_distribution(self, data, dist_type, **kwargs):
        '''
        Fit given distribution to historical precipitation data.
        The fit is accomplished using MLE or Maximum Likelihood Estimation.
        One should put some thought into the type of distribution fit to the
        data. Precipitation can have zero value and some distributions are only
        defined over interval (0, inf). Python's gamma distribution is defined
        over [0, inf). In addition SPEI which is constructed from precipitation
        - PET or (P-PET) can take on negative values!!
        
        Check https://docs.scipy.org/doc/scipy/reference/stats.html for 
        distribution types
        
        dist_type -- distribution type to fit
        data      -- Historical data to fit
        kwargs    -- scale and location parameters. See documentation on 
                     scipy.stats.rv_continuous.fit
                     
        Returns: shape, location, and scale (tuple of floats)
        '''
        
        # Get distribution type
        try:
            self.distr = getattr(scipy.stats, dist_type)
        except AttributeError:
            print ("{} is not a valid distribution type".format(dist_type))
        
        
        # Fit distribution
        params = self.distr.fit(data, **kwargs)
        
        return params
    
    
    def data_generator(self, read_func, file_list, *args):
        ''' 
        Generator to read data. Use the Generator function if you have a 
        significant amount of data that can not be read into memory at once.
        
        read_func  -- user generated function to read data
        file_list  -- list of filenames to read with read_func
        *args      -- any arguments necessary for user defined read function
        '''
        
        for fname in file_list:
            yield read_func(fname, *args)
          
           
    def calculate(self, data, starting_month=1):
        '''
        First dimension of data should be time (months)
        '''
        
        # Check if distribution has been fit on historical data
        if self.dist_type is None: 
            print ("You must fit a distribution first")
            return False
        
        if isinstance(data, types.GeneratorType):
            pass
            
        else:
            spi = self.calculate_over_full_series(data, starting_month)  
            
        return spi
           
           
    def calculate_over_full_series(self, data, starting_month):
        
        # Number of months in data
        n_months = np.shape(data)[0]
        
        # Create month list
        mnth_list = create_month_cycle(n_months, start_month=starting_month)
        
        # Pre-allocate SPI
        spi = np.zeros(np.shape(data))*np.nan
        
        # Single date series
        if data.ndim == 1:
            data = data.reshape(len(data), 1)
            spi = spi.reshape(len(spi), 1)
        
        # Loop over other series (non-time)
        for i in range(np.shape(data)[1]):
            data_one_series = np.copy(data[:,i])
            
            # Apply rolling window
            if self.window_type:
                data_one_series = self.rolling_window_mean(
                    data_one_series, self.span, self.window_type, 
                    self.rw_center, **self.rw_kwargs
                )
                
            # Loop over months
            for j in range(1,13):
                mnth_inds = np.where(mnth_list==j)[0]
                
                if len(mnth_inds)==0: 
                    continue
                
                data_month = data_one_series[mnth_inds]
                
                # Find all nans in data and remove for fitting distribution
                nan_inds = np.where(np.isnan(data_month))[0]
                data_month = data_month[~np.isnan(data_month)]
                data_month_sorted= np.sort(data_month)
                mnth_inds = np.delete(mnth_inds, nan_inds)                
                
                # Fit distribution for particular series and month
                params = self.fit_distribution(
                    data_month_sorted, self.dist_type, **self.dist_kwargs
                )
                
                # Calculate SPI/SPEI
                spi[mnth_inds,i] = self.cdf_to_ppf(data_month, params)

        return spi
    
        
    def cdf_to_ppf(self, data, params):
        '''
        Take the specific distributions fitted parameters and calculate the
        cdf. Apply the inverse normal distribution to the cdf to get the SPI 
        SPEI. This process is best described in Lloyd-Hughes and Saunders, 2002
        which is included in the documentation.
        
        '''
        
        # Calculate the CDF of observed precipitation on a given time scale
        cdf = self.distr.cdf(
            data, *params[:-2], loc=params[-2], scale=params[-1]
        )
        
        # Apply inverse normal distribution
        norm_ppf = scipy.stats.norm.ppf(cdf)
        
        return norm_ppf

    
    def best_fit_distribution(self, data, dist_list, bins=10, save_file=None):
        '''
        Calculates the Sum of the Squares error between fitted distribution and
        pdf. 
        Inspired by: http://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python
        '''
        
        y, x = np.histogram(data, bins=bins, normed=True)
        x = (x + np.roll(x, -1))[:-1] / 2.0        
        
        sse = {}
        

        fig, ax = plt.subplots()
        ax.bar(x, y, width=0.5, align='center', color='b', alpha=0.5, 
               label='data')
        
        for i,dist_name in enumerate(dist_list):
            dist = getattr(scipy.stats, dist_name)
            
            params = dist.fit(data)
            
            pdf = dist.pdf(x, *params[:-2], loc=params[-2], scale=params[-1])
            
            sse[dist_name] = np.sum((y - pdf)**2)
            
            ax.plot(x, pdf, label=dist_name)
                

        ax.legend()
        ax.grid(True)
        
        if save_file:
            plt.savefig(save_file, dpi=400)
        else:
            plt.show()
            
        return sse