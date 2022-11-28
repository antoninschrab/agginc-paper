from __future__ import absolute_import, print_function

import numpy as np
import scipy

from kgof.density import UnnormalizedDensity, Normal

from .data import DSMultivariateStudentsT, DSSech
from .distributions import multivariate_iso_t_logpdf


class MultivariateStudentsT(UnnormalizedDensity):
    def __init__(self, m, var, df):
        self.m = m
        self.var = var
        self.df = df

    def log_den(self, X):
        return multivariate_iso_t_logpdf(X, self.m, self.var, self.df)

    def log_normalized_den(self, X):
        return self.log_den(X)

    def get_datasource(self):
        return DSMultivariateStudentsT(self.m, self.var, self.df)

    def dim(self):
        return len(self.m)



class Sech(UnnormalizedDensity):
    def __init__(self, mean, scale):
        self.mean = mean
        self.scale = scale

    def log_den(self, X):
        X_normalized = np.sqrt(np.pi/2) * self.scale * (X - self.mean)
        return np.sum(np.logaddexp(X_normalized, -X_normalized), 1)

    def log_normalized_den(self, X):
        Z = self.scale / np.sqrt(np.pi/2) / 2.
        return self.log_den(X) + self.mean.shape[0] * np.log(Z)

    def get_datasource(self):
        return DSSech(self.mean, self.scale)

    def dim(self):
        return len(self.mean)


class Normal2(Normal):
    def log_normalized_den(self, X):
        return stats.multivariate_normal.logpdf(X, mean=self.mean, cov=self.cov)
