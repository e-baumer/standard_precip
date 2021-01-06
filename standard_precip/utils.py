import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_index(df: pd.DataFrame, date_col: str, precip_col: str, save_file: str=None,
               index_type: str='SPI', bin_width: int=22):

    pos_index = df.loc[df[precip_col] >= 0]
    neg_index = df.loc[df[precip_col] < 0]

    fig, ax = plt.subplots()
    ax.bar(pos_index[date_col], pos_index[precip_col], width=bin_width, align='center', color='b')
    ax.bar(neg_index[date_col], neg_index[precip_col], width=bin_width, align='center', color='r')
    ax.grid(True)
    ax.set_xlabel("Date")
    ax.set_ylabel(index_type)

    if save_file:
        plt.savefig(save_file, dpi=400)

    return fig

def best_fit_distribution(data: np.array, dist_list: list, fit_type: str='lmom', bins: int=10,
                          save_file: str=None, **kwargs):
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

    for dist_name in dist_list:
        distrb = getattr(distr, dist_name)

        if fit_type == 'lmom':
            params = distrb.lmom_fit(data, **kwargs)

        elif fit_type == 'mle':
            params = distrb.fit(data, **kwargs)

        else:
            raise AttributeError(f"{fit_type} is not an option. Option fit_types are mle and lmom")

        pdf = distrb.pdf(x, **params)
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

