from __future__ import absolute_import, print_function

import numpy as np
import scipy.stats as stats
import scipy.special as special
from scipy.stats._multivariate import _PSD


def rff_cauchy_sampler(c, d):
    def rff_sampler(n):
        return np.random.laplace(scale=1./c, size=(n, d))
    return rff_sampler


def multipoint_rvs(l1, l2, p=.5, n=1):
    choices = np.array([l1,l2])
    return np.random.choice(choices, size=n, p=[p, 1-p])


def spherical_rvs(d, r, n=1):
    X = np.random.randn(n, d)
    return r * (X / np.linalg.norm(X, axis=1)[:,np.newaxis])


def sech_rvs(m, scale, n=1):
    d = m.shape[-1]
    c = 2 * np.sqrt(2) / scale / np.pi**1.5
    x = c * np.log(np.tan(np.pi / 2 * np.random.random((n, d)))) + m
    return x


def sech_logpdf(x, m, scale):
    d = m.shape[-1]
    x_normalized = np.sqrt(np.pi/2) * scale * (x - m)
    logpdfs = np.logaddexp(x_normalized, -x_normalized)
    if d > 1:
        logpdfs = np.sum(logpdfs, -1)
    Z = scale / np.sqrt(np.pi/2) / 2
    return logpdfs.squeeze() + d * np.log(Z)


def multivariate_t_rvs(m, S, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    S : array_like
        square array of covariance  matrix
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = m.shape[-1]
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.multivariate_normal(np.zeros(d),S,(n,))
    return m + z/np.sqrt(x)[:,None]   # same output format as random.multivariate_normal

def multivariate_iso_t_rvs(m, var, df=np.inf, n=1):
    '''generate random variables of multivariate t distribution

    Parameters
    ----------
    m : array_like
        mean of random variable, length determines dimension of random variable
    var : float
        variance
    df : int or float
        degrees of freedom
    n : int
        number of observations, return random array will be (n, len(m))
    Returns
    -------
    rvs : ndarray, (n, len(m))
        each row is an independent draw of a multivariate t distributed
        random variable
    '''
    m = np.asarray(m)
    d = m.shape[-1]
    if df == np.inf:
        x = np.ones(1)
    else:
        x = np.random.chisquare(df, n)/df
    z = np.random.randn(n,d)
    rvs = m + z/np.sqrt(x)[:,np.newaxis]   # same output format as random.multivariate_normal
    return rvs

# See: https://github.com/scipy/scipy/blob/master/scipy/stats/_multivariate.py
def multivariate_t_logpdf(x, m, S, df=np.inf):
    """calculate log pdf for each value

    Parameters
    ----------
    x : array_like, shape=(n_samples, n_features)

    m : array_like, shape=(n_features,)

    S : array_like, shape=(n_features, n_features)
        covariance  matrix
    df : int or float
        degrees of freedom
    """
    m = np.asarray(m)
    d = m.shape[-1]
    if df == np.inf:
        return stats.multivariate_normal.logpdf(x, m, S)
    psd = _PSD(S)

    log_pdf = special.gammaln(.5*(df + d)) - special.gammaln(.5*df) - .5*d * np.log(np.pi * df)
    log_pdf += -.5*psd.log_pdet
    dev = x - m
    maha = np.sum(np.square(np.dot(dev, psd.U)), axis=-1)
    log_pdf += -.5*(df + d) * np.log(1 + maha / df)
    return log_pdf


def multivariate_iso_t_logpdf(x, m, var, df=np.inf):
    """calculate log pdf for each value

    Parameters
    ----------
    x : array_like, shape=(n_samples, n_features)

    m : array_like, shape=(n_features,)

    var : float
        variance
    df : int or float
        degrees of freedom
    """
    m = np.asarray(m)
    d = m.shape[-1]
    if df == np.inf:
        return -.5 * np.sum((x - m)**2, axis=-1) / var - .5 * d * np.log(np.pi * 2 * var)

    log_pdf = special.gammaln(.5*(df + d)) - special.gammaln(.5*df) - .5*d * np.log(np.pi * df * var)
    maha = np.sum((x - m)**2, axis=-1) / var
    log_pdf += -.5*(df + d) * np.log(1 + maha / df)
    return log_pdf
