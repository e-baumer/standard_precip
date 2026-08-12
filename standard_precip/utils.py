import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from standard_precip import _distributions


def plot_index(
    df: pd.DataFrame,
    date_col: str,
    precip_col: str,
    save_file: str | None = None,
    index_type: str = "SPI",
    bin_width: int = 22,
):
    """
    Plot a calculated index as a bar chart, with positive (wet) values in blue and
    negative (dry) values in red.

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe returned by calculate(), containing the date and index columns.

    date_col: str
        Name of the date column.

    precip_col: str
        Name of the calculated index column to plot (e.g. 'precip_calculated_index').

    save_file: str, optional
        File path to save the figure. If not given, the figure is only returned.

    index_type: str (default='SPI')
        Label for the y-axis.

    bin_width: int (default=22)
        Bar width in days. The default suits monthly data; use ~1 for daily data.

    Returns
    -------
    fig: matplotlib.figure.Figure
    """
    pos_index = df.loc[df[precip_col] >= 0]
    neg_index = df.loc[df[precip_col] < 0]

    fig, ax = plt.subplots()
    ax.bar(pos_index[date_col], pos_index[precip_col], width=bin_width, align="center", color="b")
    ax.bar(neg_index[date_col], neg_index[precip_col], width=bin_width, align="center", color="r")
    ax.grid(True)
    ax.set_xlabel("Date")
    ax.set_ylabel(index_type)

    if save_file:
        plt.savefig(save_file, dpi=400)

    return fig


def best_fit_distribution(
    data: np.ndarray,
    dist_list: list,
    fit_type: str = "lmom",
    bins: int = 10,
    save_file: str | None = None,
    **kwargs,
):
    """
    Method to find the best distribution for observational data. Calculates the Sum of the
    Squares error between fitted distribution and pdf.
    Inspired by: http://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python

    Each candidate is fit exactly the way calculate() fits it: for distributions that are
    undefined at zero (gamma, Pearson III) the zero observations are removed before fitting
    and the fitted density is weighted by (1 - p_zero) per the mixed CDF of Thom (1966),
    and gamma MLE fixes loc=0 unless a location constraint is passed - so the selected
    distribution corresponds to the model the index calculation will actually use.

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
    """
    data = np.asarray(data)
    p_zero = float(np.mean(data == 0)) if data.size else 0.0

    y, x = np.histogram(data, bins=bins, density=True)
    x = (x + np.roll(x, -1))[:-1] / 2.0

    sse = {}
    fig, ax = plt.subplots()
    ax.bar(x, y, width=0.5, align="center", color="b", alpha=0.5, label="data")

    for dist_name in dist_list:
        spec = _distributions.get_spec(dist_name)
        fit_data = data[data != 0] if spec.strip_zeros else data
        distrb, params = _distributions.fit(spec, fit_data, fit_type, **kwargs)

        pdf = distrb.pdf(x, **params)
        if spec.strip_zeros:
            pdf = (1 - p_zero) * pdf
        sse[dist_name] = np.sum((y - pdf) ** 2)
        ax.plot(x, pdf, label=dist_name)

    ax.legend()
    ax.grid(True)

    if save_file:
        plt.savefig(save_file, dpi=400)
    else:
        plt.show()

    sse = sorted(sse.items(), key=lambda x: x[1], reverse=False)
    return sse
