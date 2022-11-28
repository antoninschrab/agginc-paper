from __future__ import absolute_import, print_function


from kgof.util import NumpySeedContext
from kgof.data import Data, DataSource

import numpy as np
import scipy

from .distributions import multivariate_iso_t_rvs


class DSStudentsT(DataSource):
    def __init__(self, d, df):
        self.d = d
        self.df = df

    def sample(self, n, seed=None):
        with NumpySeedContext(seed=seed):
            X = np.random.standard_t(self.df, size=(n, self.d))
            return Data(X)


class DSMultivariateStudentsT(DataSource):
    def __init__(self, m, var, df):
        self.m = m
        self.var = var
        self.df = df

    def sample(self, n, seed=None):
        with NumpySeedContext(seed=seed):
            X = multivariate_iso_t_rvs(self.m, self.var, self.df, n)
            return Data(X)


class DSSech(DataSource):
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale

    def sample(self, n, seed=None):
        """Generate random variable from pdf proportional to sech(sqrt(pi/2) scale (x - mean))

        Has mean mean and variance 2 / (pi * scale^2)"""
        with NumpySeedContext(seed=seed):
            d = self.mean.shape[0]
            c = 2 * np.sqrt(2) / self.scale / np.pi**1.5
            X = c * np.log(np.tan(np.pi / 2 * np.random.random((n, d)))) + self.mean
            return Data(X)
