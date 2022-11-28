from __future__ import absolute_import, print_function

import errno
import os
import time
import pickle

import autograd.numpy as np
from scipy import stats
from scipy import misc
from scipy import special
import sklearn.metrics as metrics

from abc import ABCMeta, abstractmethod


def store_objects(_object_store_loc, **kwargs):
    create_folder_if_not_exist(_object_store_loc)
    for k, v in kwargs.items():
        with open(os.path.join(_object_store_loc, k + '.pck'), 'wb') as f:
            pickle.dump(v, f)


def restore_object(_object_store_loc, name):
    with open(os.path.join(_object_store_loc, name + '.pck'), 'rb') as f:
        return pickle.load(f) #, encoding='bytes')


def format_seconds(secs):
    if secs < 1e-3:
        t, u = secs * 1e6, 'microsec'
    elif secs < 1e0:
        t, u = secs * 1e3, 'millisec'
    else:
        t, u = secs, 'sec'
    return '%.03f %s' % (t, u)


class Timer:
    def __init__(self, descr=None):
        self.description = descr

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        if self.description is not None:
            time_str = format_seconds(self.interval)
            print('%s took %s to run' % (self.description, time_str))


def create_folder_if_not_exist(path):
    # create the output folder if it doesn't exist
    try:
        os.makedirs(path)
        print('Created output folder:', path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            print('Unknown error creating output directory', path)
            raise


def pretty_file_string_from_dict(d):
    if len(d) == 0:
        return ''
    keys = list(d.keys())
    keys.sort()
    s = '-'.join(['%s=%s' % (k, d[k]) for k in keys if not callable(d[k])])
    return s


def nice_str(o):
    if isinstance(o, list):
        return ','.join(map(nice_str, o))
    else:
        return str(o)

class SLArray(object):
    def __init__(self, log_a, signs=1):
        log_a = np.asarray(log_a)
        signs = np.asarray(signs)
        if signs.size > 1 and log_a.shape != signs.shape:
            raise ValueError('log_a and signs have incompatible shapes')
        self._log_a = log_a
        self._signs = signs

    def _set_from_array(self, a):
        a = np.asarray(a)
        np.seterr(divide='ignore')
        if a.ndim == 0 and a == 0:
            self._log_a = np.asarray(-np.inf)
            self.signs = np.array(1)
        if np.all(a >= 0):
            self._log_a = np.log(a)
            if a.ndim > 0:
                self._log_a[a == 0] = -np.inf
            self._signs = np.array(1)
        else:
            self._log_a = np.log(np.abs(a))
            if a.ndim > 0:
                self._log_a[a == 0] = -np.inf
            self._signs = np.sign(a)
        np.seterr(divide='warn')

    def _as_rescaled_array(self, axis=None):
        max_values = self._log_a.max(axis)
        return self._signs * np.exp(self._log_a - max_values), max_values

    def _set_to_rescaled_array(self, a, max_values):
        self._set_from_array(a)
        self._log_a += max_values

    ## Class methods ##

    @classmethod
    def from_array(cls, a):
        """Construct an SLArray object equivalent to an array-like object"""
        obj = cls(0)
        obj._set_from_array(a)
        return obj

    @classmethod
    def as_slarray(cls, a):
        if isinstance(a, cls):
            return a
        else:
            return cls.from_array(a)

    ## Properties ##

    @property
    def ndim(self):
        return self._log_a.ndim

    @property
    def size(self):
        return self._log_a.size

    @property
    def shape(self):
        return self._log_a.shape

    @property
    def T(self):
        if self.ndim < 2:
            return self
        else:
            return SLArray(self._log_a.T, self._signs.T)

    ## Get and set slices ##

    def __getitem__(self, sl):
        log_a = self._log_a.__getitem__(sl)
        if self._signs.ndim == 0:
            signs = self._signs.copy()
        else:
            signs = self._signs.__getitem__(sl)
        return SLArray(log_a, signs)

    def __setitem__(self, sl, val):
        val = SLArray.as_slarray(val)
        self._log_a.__setitem__(sl, val._log_a)
        if (val._signs.ndim > 0):
            if self._signs.ndim == 0:
                self._signs = self._signs * np.ones_like(self._log_a)
        # XXX: finish implementing



    ## Manipulations ##

    def min(self, axis=None):
        a, max_values = self._as_rescaled_array(axis)
        obj = SLArray(0)
        obj._set_to_rescaled_array(a.min(axis), max_values)
        return obj

    def max(self, axis=None):
        a, max_values = self._as_rescaled_array(axis)
        obj = SLArray(0)
        obj._set_to_rescaled_array(a.max(axis), max_values)
        return obj

    def mean(self, axis=None, dtype=None, out=None):
        s = self.sum(axis)
        if axis is None:
            count = self.size
        else:
            count = self.shape[axis]
        s._log_a -= np.log(count)
        return s

    def sum(self, axis=None, dtype=None, out=None):
        logsums, signs = signed_logsumexp(self._log_a, self._signs, axis=axis)
        return SLArray(logsums, signs)

    def sqrt(self):
        if np.any(self._signs < 0):
            raise RuntimeError('invalid value encountered in sqrt')
        return SLArray(self._log_a / 2.)

    def abs(self):
        return SLArray(self._log_a)

    def diagonal(self, **kwargs):
        log_a = self._log_a.diagonal(**kwargs)
        if self._signs.ndim == 0:
            signs = self._signs.copy()
        else:
            signs = self._signs.diagonal(**kwargs)
        return SLArray(log_a, signs)

    def reshape(self, newshape):
        log_a = self._log_a.reshape(newshape)
        if self._signs.ndim == 0:
            signs = self._signs
        else:
            signs = self._signs.reshape(newshape)
        return SLArray(log_a, signs)

    def toarray(self):
        return self._signs * np.exp(self._log_a)


    # def dot(self, other):
    #     other = SLArray.as_slarray(other)
    #     max_value = max(np.max(self._log_a), np.max(other._log_a))
    #     A = self._signs * np.exp(self._log_a - max_value)
    #     B = other._signs * np.exp(other._log_a - max_value)
    #     scaled_output = A.dot(B)
    #     slarray_output = SLArray.from_array(scaled_output)
    #     slarray_output._log_a += 2*max_value
    #     return slarray_output

    def dot(self, other):
        other = SLArray.as_slarray(other)
        if self.ndim > 0:
            A_row_max = np.max(self._log_a, axis=-1, keepdims=True)
        else:
            A_row_max = self._log_a
        if other.ndim > 0:
            B_col_max = np.max(other._log_a, axis=0, keepdims=True)
        else:
            B_col_max = other._log_a
        A = self._signs * np.exp(self._log_a - A_row_max)
        B = other._signs * np.exp(other._log_a - B_col_max)
        scaled_output = A.dot(B)
        slarray_output = SLArray.from_array(scaled_output)
        slarray_output._log_a += (A_row_max + B_col_max).reshape(slarray_output.shape)
        return slarray_output

    ## Basic Operations ##

    # def rescale(self, scales):
    #     assert self.log_scale == scales.log_scale
    #     if self.log_scale:
    #         self._log_a += scales._log_a
    #         self._signs *= scales._signs
    #     else:
    #         self._log_a *= scales._log_a

    def __truediv__(self, other):
        other = SLArray.as_slarray(other)
        new_log_a = self._log_a - other._log_a
        new_signs = self._signs * other._signs
        return SLArray(new_log_a, new_signs)

    def __div__(self, other):
        return self.__truediv__(other)

    def __mul__(self, other):
        other = SLArray.as_slarray(other)
        new_log_a = self._log_a + other._log_a
        new_signs = self._signs * other._signs
        return SLArray(new_log_a, new_signs)

    def __imul__(self, other):
        other = SLArray.as_slarray(other)
        self._log_a += other._log_a
        self._signs *= other._signs
        return self

    def __add__(self, other):
        other = SLArray.as_slarray(other)
        stacked_log_a = np.stack((self._log_a, other._log_a))
        stacked_signs = np.stack((self._signs, other._signs))
        new_log_a, new_signs = signed_logsumexp(stacked_log_a, stacked_signs,
                                                axis=0)
        return SLArray(new_log_a, new_signs)

    def __sub__(self, other):
        other = SLArray.as_slarray(other)
        stacked_log_a = np.stack((self._log_a, other._log_a))
        stacked_signs = np.stack((self._signs, -other._signs))
        new_log_a, new_signs = signed_logsumexp(stacked_log_a, stacked_signs,
                                                axis=0)
        return SLArray(new_log_a, new_signs)

    def __pow__(self, other):
        if isinstance(other, SLArray):
            other = other.toarray()
        new_log_a = self._log_a * other
        new_signs = self._signs ** other
        return SLArray(new_log_a, new_signs)

    def __str__(self):
        #return '%s(tensor=%s, signs=%s)' % (self.__class__, self._log_a, self._signs)
        return str(self.toarray())


# based on numpy.linalg.norm
def log_scale_vector_norm(x, ord=None, axis=None):
    """
    Computes log(||exp(x)||_ord) along a single axis

    """
    x = np.asarray(x)

    if not issubclass(x.dtype.type, (np.inexact, np.object_)):
        x = x.astype(float)

    # Normalize the `axis` argument to a tuple.
    nd = x.ndim
    if axis is None:
        axis = 0
        x = x.ravel(order='K')
    elif not isinstance(axis, int):
        raise TypeError("'axis' must an integer")

    if ord is None:
        ord = 2

    if ord == np.Inf:
        return x.max(axis=axis)
    elif ord == -np.Inf:
        return x.min(axis=axis)
    elif ord == 0:
        # Zero norm
        return np.log((x != -np.Inf).astype(float).sum(axis=axis))
    elif ord == 1:
        # special case for speedup
        return special.logsumexp(x, axis=axis)
    else:
        try:
            ord + 1
        except TypeError:
            raise ValueError("Invalid norm order for vectors.")
        return special.logsumexp(ord * x, axis=axis) / float(ord)


# based on scipy.special.logsumexp
def signed_logsumexp(a, signs, axis=None):
    """
    Compute the log of the sum of exponentials of input elements using sign
    information.
    """
    a = np.asarray(a)
    signs = np.asarray(signs)
    if axis is None:
        a = a.ravel()
        signs = signs.ravel()
    else:
        a = np.rollaxis(a, axis)
        if signs.ndim > 0:
            signs = np.rollaxis(signs, axis)
    a_max = a.max(axis=0)
    sums = np.sum(signs * np.exp(a - a_max), axis=0)
    sums_signs = np.sign(sums)
    # sums[sums_signs == 0] = -np.inf
    np.seterr(divide='ignore')
    out = np.log(np.abs(sums))
    np.seterr(divide='warn')
    out += a_max
    if len(out.shape) == 0:
        if sums_signs == 0:
            out = -np.inf
    else:
        out[sums_signs == 0] = -np.inf

    return out, sums_signs


def qmean(X, **kwargs):
    """
    Compute the quadratic mean of the data
    """
    return np.sqrt(np.mean(X**2, **kwargs))


def meddistance(X, subsample=None, mean_on_fail=True, metric='euclidean'):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.

    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and
        there are more slightly more 0 than 1.

    Return
    ------
    median distance
    """
    if subsample is None:
        D = metrics.pairwise_distances(X, metric=metric)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med
    else:
        assert subsample > 0
        rand_state = np.random.get_state()
        np.random.seed(None)
        n = X.shape[0]
        ind = np.random.choice(n, min(subsample, n), replace=False)
        np.random.set_state(rand_state)
        # recursion just one
        return meddistance(X[ind, :], None, mean_on_fail, metric)


def binomial_interval(inter, k, n):
    """Calculate (100*inter)%% confidence interval for binomial parameter

    k : number of successes
    n : number of trials
    """
    tail_prob = (1 - inter) / 2.
    if k == 0:
        lower = 0
    else:
        lower = stats.beta.isf(1 - tail_prob, k + .5, n - k +.5)
    if k == n:
        upper = 1
    else:
        upper = stats.beta.isf(tail_prob, k + .5, n - k +.5)
    return (lower, upper)
