from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import stats
from scipy import special
from scipy import integrate
from scipy import misc

from kgof.data import DataSource
from kgof.density import UnnormalizedDensity, Normal
from kgof.goftest import FSSD
from kgof.kernel import DifferentiableKernel, KSTKernel

from .distributions import multivariate_iso_t_rvs, multivariate_iso_t_logpdf
from .distributions import sech_rvs, sech_logpdf
from .kernel import (KSechMult, SmoothedExpMultiplier, UnityMultiplier,
                   PolynomialMultiplier, KIMQ2, KGauss2)
from .util import log_scale_vector_norm, signed_logsumexp, SLArray


class Divergence(object):
    """Abstract class representing a divergence measure between distributions
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass

    @abstractmethod
    def divergence(self, X, Y=None, **kwargs):
        """Calculate divergence between samples (or a fixed distribution)"""
        raise NotImplementedError()

    def scaled_divergence(self, X, Y=None, **kwargs):
        """Scaled version of the divergence (if available)

        By default the same as divergence"""
        return self.divergence(X, Y, **kwargs)


class KSD(Divergence):
    def __init__(self, k, p):
        super(KSD, self).__init__()
        self.k = k
        self.p = p

    def divergence(self, X, Y=None, **kwargs):
        """
        Compute the V statistic as in Section 2.2 of Chwialkowski et al., 2016.
        """
        k = self.k
        p = self.p
        n, d = X.shape
        # n x d matrix of gradients
        grad_logp = p.grad_log(X)
        # n x n
        gram_glogp = grad_logp.dot(grad_logp.T)
        # n x n
        K = k.eval(X, X)

        B = np.zeros((n, n))
        C = np.zeros((n, n))
        for i in range(d):
            grad_logp_i = grad_logp[:, i]
            B += k.gradX_Y(X, X, i)*grad_logp_i
            C += (k.gradY_X(X, X, i).T * grad_logp_i).T

        H = K*gram_glogp + B + C + k.gradXY_sum(X, X)
        # V-statistic
        ksd = np.sqrt(np.mean(H))
        return ksd

    def __str__(self):
        return '%s(k=%s)' % (type(self).__name__, self.k)


def _validate_ordering(ordering):
    if ordering not in ['ij', 'ji']:
        raise ValueError('invalid ordering "%s"' % ordering)


def _construct_order(order):
    order = np.array(order, ndmin=1)
    if len(order) == 1:
        order = np.append(order, 2)
    if len(order) != 2:
        raise ValueError('order must be an int, float or array-like of length 1 or 2')
    return order


class RandomFeatureDivergence(Divergence):
    def __init__(self, order, sampler, log_density, ordering):
        """
        Parameters
        ----------
        order : int, float, or array-like
            Orders of the norms to use. If only one value given it is the
            order of the norm along each dimension and the 2-norm is used
            across features. If two values given, they are
            [order-across-dimensions, order-across-features]

        sampler : function

        log_density : function

        ordering : str
            Ordering of norms.
                'ij' = take the norms along each dimension first (sum over i on outside)
                'ji' = take the norms along each feature first (sum over j on outside)
        """
        _validate_ordering(ordering)
        super(RandomFeatureDivergence, self).__init__()
        self.order = _construct_order(order)
        self.sampler = sampler
        self.log_density = log_density
        self.ordering = ordering
        self.Z = None
        self.data_mean = None

    @abstractmethod
    def log_scale(self):
        """Return whether the feature tensor is on the log scale"""
        raise NotImplementedError()

    @abstractmethod
    def _feature_tensor(self, X, Z):
        """Return a feature tensor

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_dims)
            The set of samples

        Z : array-like, shape=(n_features, n_feature_dims)
            The random features used to construct the feature tensor

        Returns
        -------
        feature_tensor : SLArray object
        """
        raise NotImplementedError()

    def _generate_features(self, data_mean, X, J, Z, reuse_features):
        # recompute features if not available, J has changed, or we are not
        # reusing features
        if (self.Z is None or
            self.Z.shape[0] != J or
            not reuse_features):
            if Z is None:
                self.Z = self.sampler(data_mean, X, J)
            else:
                self.Z = np.asarray(Z)
                J = self.Z.shape[0]
            self.Z_log_probs = self.log_density(self.Z, data_mean, X)
            log_feature_weights = -(self.Z_log_probs + np.log(J)) / self.order[0]
            if self.log_scale():
                self.feature_weights = SLArray(log_feature_weights, 1)
            else:
                self.feature_weights = np.exp(log_feature_weights)
            self.data_mean = data_mean
        return self.Z, self.feature_weights

    def compute_divergence(self, mean_weighted_features_tensor,
                           order=None, ordering=None):
        """Compute the divergence using the mean weigthed feature tensor

        Parameters
        ----------
        mean_weighted_features_tensor : SLArray
            The expected value of the weighted features

        order : float or pair of floats

        ordering : str
            Either None, 'ij' or 'ji'.

        Returns
        -------
        divergence : float
            The (log) divergence
        """
        if ordering is None:
            ordering = self.ordering
        _validate_ordering(ordering)
        if order is None:
            order = self.order
        order = _construct_order(order)
        if ordering == 'ij':
            first_axis = -1
        else:
            first_axis = -2
        # compute norm along each dimension or feature
        if self.log_scale():
            # TODO: implement norm that takes SLArray as argument
            a = log_scale_vector_norm(mean_weighted_features_tensor._log_a,
                                      order[-1-first_axis], axis=first_axis)
            a = log_scale_vector_norm(a, order[2+first_axis], axis=-1)
        else:
            a = np.linalg.norm(mean_weighted_features._log_a,
                               order[-1-first_axis], axis=first_axis)
            a = np.linalg.norm(a, order[2+first_axis], axis=-1)
        return a

    def divergence(self, X, Y=None, J=10, Z=None, reuse_features=False, ordering=None,
                   return_feature_info=False):
        data_mean = np.mean(X, axis=0)
        Z, feature_weights = self._generate_features(data_mean, X, J, Z,
                                                     reuse_features)
        # average over data
        feature_tensor = self._feature_tensor(X, Z)
        mean_feature_tensor = feature_tensor.mean(axis=0)

        if Y is not None:
            feature_tensor_Y = self._feature_tensor(Y, Z)
            mean_feature_tensor_Y = feature_tensor_Y.mean(axis=0)
            mean_feature_tensor -= mean_feature_tensor_Y
        mean_feature_tensor *= feature_weights
        divergence = self.compute_divergence(mean_feature_tensor,
                                             ordering=ordering)
        if return_feature_info:
            return divergence, feature_tensor, feature_weights
        else:
            return divergence


def _make_sampler_for_rff_ksd(rff_sampler):
    def sampler(dm, X, J):
        omega = rff_sampler(J)
        b = np.random.rand(J, 1)
        return np.concatenate((omega, b), axis=1)
    return sampler

def _log_sign_rep(a):
    return np.log(np.abs(a)), np.sign(a)

class RFFKSD(RandomFeatureDivergence):
    def __init__(self, rff_sampler, p):
        super(RFFKSD, self).__init__(2, _make_sampler_for_rff_ksd(rff_sampler),
                                     lambda Z, b, c: 0.0, 'ij')
        self.p = p

    def log_scale(self):
        return True

    def _feature_tensor(self, X, Z):
        J = Z.shape[0]
        n, d = X.shape
        # d x J
        omega = Z[:,:-1].T
        # length J vector
        b = Z[:,-1]
        # n x J
        xomegab = X.dot(omega) + b[np.newaxis, :]

        log_cos_xomegab, cos_xomegab_signs = _log_sign_rep(np.sqrt(2)*np.cos(xomegab))
        log_sin_xomegab, sin_xomegab_signs = _log_sign_rep(np.sqrt(2)*np.sin(xomegab))
        # n x d
        log_grad_logp, grad_logp_signs = _log_sign_rep(self.p.grad_log(X))
        # d x J
        log_omega, omega_signs = _log_sign_rep(omega)

        # n x d x J tensors
        log_cos_grad_log_p = log_cos_xomegab[:,np.newaxis,:] + log_grad_logp[:,:,np.newaxis]
        cos_grad_log_p_signs = cos_xomegab_signs[:,np.newaxis,:] * grad_logp_signs[:,:,np.newaxis]
        log_sin_xomegab_omega = log_sin_xomegab[:,np.newaxis,:] + log_omega[np.newaxis,:,:]
        sin_xomegab_omega_signs = sin_xomegab_signs[:,np.newaxis,:] * omega_signs[np.newaxis,:,:]

        stacked_tensors = np.stack((log_cos_grad_log_p, log_sin_xomegab_omega))
        stacked_signs = np.stack((cos_grad_log_p_signs, -sin_xomegab_omega_signs))
        log_Xi, Xi_signs = signed_logsumexp(stacked_tensors, stacked_signs,
                                            axis=0)
        assert log_Xi.shape == (n, d, J)
        assert log_Xi.shape == Xi_signs.shape
        return SLArray(log_Xi, Xi_signs)



class FastMMD(RandomFeatureDivergence):
    def __init__(self, order, sampler, log_density, f, log_scale=False):
        super(FastMMD, self).__init__(order, sampler, log_density, 'ij')
        self.f = f
        self._log_scale = log_scale

    def log_scale(self):
        return self._log_scale

    def _feature_tensor(self, X, Z):
        if self.log_scale():
            return self.f.log_eval(X, Z)
        else:
            return self.f.eval(X, Z)


class FastKSD(RandomFeatureDivergence):
    def __init__(self, order, sampler, log_density, f, p, ordering='ij'):
        super(FastKSD, self).__init__(order, sampler, log_density, ordering)
        self.f = f
        self.p = p

    def log_scale(self):
        return True

    def _feature_tensor(self, X, Z):
        """
        Compute the feature tensor which is n x d x J.
        """
        J = Z.shape[0]
        n, d = X.shape
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        grad_logp_signs = np.sign(grad_logp)
        log_grad_logp = np.log(np.abs(grad_logp))
        # print('log grad logp =', log_grad_logp)
        # print('grad logp signs =', grad_logp_signs)
        # n x J matrix
        logK = self.f.log_eval(X, Z)
        # print('log K =', logK)
        # print('exp log K =', np.exp(logK))
        # print('K =', self.f.eval(X, Z))

        list_grads = []
        list_signs = []
        for z in Z:
            g, s = self.f.log_gradX_y(X, z)
            list_grads.append(np.reshape(g, (1, n, d)))
            list_signs.append(np.reshape(s, (1, n, d)))
        log_fgrads = np.transpose(np.concatenate(list_grads, axis=0), (1, 2, 0))
        log_fgrads_signs = np.transpose(np.concatenate(list_signs, axis=0),
                                        (1, 2, 0))
        # print('fgrads =', SLArray(log_fgrads, log_fgrads_signs))
        # print('log fgrads =', log_fgrads)
        # print('log fgrads signs =', log_fgrads_signs)

        # n x d x J tensor
        log_grad_logp_K = log_grad_logp[:,:,np.newaxis] + logK[:,np.newaxis,:]
        log_grad_logp_K_signs = np.tile(grad_logp_signs[:,:,np.newaxis], (1, 1, J))
        stacked_tensors = np.stack((log_grad_logp_K, log_fgrads))
        stacked_signs = np.stack((log_grad_logp_K_signs, log_fgrads_signs))
        log_Xi, Xi_signs = signed_logsumexp(stacked_tensors, stacked_signs,
                                            axis=0)
        assert log_Xi.shape == (n, d, J)
        assert log_Xi.shape == Xi_signs.shape
        return SLArray(log_Xi, Xi_signs)


class L1IMQFastKSD(FastKSD):
    def __init__(self, p, c=1, beta=-0.5, gamma=.25, d=1, ordering='ij',
                 other_order=2, nu_type='mean', target_df=3):
        f1, sampler, log_density, c_prime, beta_prime = \
            self._generate_imq_f1(c, beta, d, gamma, nu_type, float(target_df))
        super(L1IMQFastKSD, self).__init__([1, other_order], sampler,
                                           log_density, f1, p, ordering)
        self.beta_prime = beta_prime
        self.c_prime = c_prime
        self.c = c
        self.beta = beta
        self.gamma = gamma
        self.d = d

    def _generate_imq_f1(self, c, beta, d, gamma, nu_type, target_df):
        """
        c and beta are the IMQ parameters
        d is the dimensionality
        gamma determines the required sample size: m >= O(\bar Y^{-gamma})
        """
        # XXX temporary
        # xi = gamma / 2.
        # xi_min = .95 * xi  # the .95 here is arbitrary
        # s_max = 1 - gamma / 4.
        # beta_prime = -d / (2 * xi_min)
        # c_prime = s_max * c / 2.
        # p = 1
        # r = 2

        # XXX original
        alpha = gamma / 3.
        xi = 4 * alpha / (2 + alpha)
        xi_min = d / (d + target_df) * xi
        lambda_max = 1 - alpha / 2.
        beta_prime = -d / (2 * xi_min)
        c_prime = lambda_max * c / 2.
        p = 1

        f1 = KIMQ2(beta_prime, c_prime)
        # p * xi * beta' = -(df + d)/2
        nu_df = -2 * p * xi * beta_prime - d
        nu_sigma2 = c_prime**2 / nu_df

        if nu_type == 'mean':
            nu_sampler = lambda mean, X, n: multivariate_iso_t_rvs(mean, nu_sigma2, nu_df, n)
            nu_log_density = lambda x, mean, X: multivariate_iso_t_logpdf(x, mean, nu_sigma2, nu_df)
        elif nu_type == 'mixture':
            nu_df = np.inf
            nu_sigma = 4 / d
            def nu_sampler(mean, X, n):
                grad_log_p = self.p.grad_log(X)
                wts = np.linalg.norm(grad_log_p, axis=1)
                wts /= np.sum(wts)
                inds = np.random.choice(wts.size, size=n, p=wts)
                return multivariate_iso_t_rvs(X[inds], nu_sigma2, nu_df, n)
            def nu_log_density (x, mean, X):
                lls = multivariate_iso_t_logpdf(x[:,np.newaxis,:], X[np.newaxis,:,:], nu_sigma2, nu_df)
                grad_log_p = self.p.grad_log(X)
                wts = np.linalg.norm(grad_log_p, axis=1)
                lls += np.log(wts)
                return misc.logsumexp(lls, axis=1) - np.log(np.sum(wts))
        elif nu_type == 'non-isotropic':
            def nu_sampler(mean, X, n):
                data_mean = np.mean(X, 0)
                data_cov = np.cov(X.T) + 1e-5*np.eye(data_mean.size)
                return stats.multivariate_normal.rvs(data_mean, data_cov, size=n)
            def nu_log_density (x, mean, X):
                data_mean = np.mean(X, 0)
                data_cov = np.cov(X.T) + 1e-5*np.eye(data_mean.size)
                return stats.multivariate_normal.logpdf(x, data_mean, data_cov)
        else:
            raise RuntimeError('invalid nu type')

        return f1, nu_sampler, nu_log_density, c_prime, beta_prime

    def divergence(self, X, Y=None, **kwargs):
        if X.shape[1] != self.d or (Y is not None and Y.shape[1] != self.d):
            raise ValueError('X (or Y) has the wrong dimension')
        return super(L1IMQFastKSD, self).divergence(X, Y, **kwargs)

    def __str__(self):
        return '%s(c=%f, beta=%f, gamma=%f, d=%d)' % (type(self).__name__,
                                                      self.c, self.beta,
                                                      self.gamma, self.d)



class LrSechFastKSD(FastKSD):
    def __init__(self, p, r=2, scale=1, gamma=.25, multiplier='exp', nu_type='mean'):
        sqrt_k, sampler, log_density = self._generate_sqrt_sech(r, scale, gamma,
                                                                multiplier,
                                                                nu_type)
        super(LrSechFastKSD, self).__init__(r, sampler, log_density, sqrt_k, p)
        self.scale = scale
        self.gamma = gamma

    def _generate_sqrt_sech(self, r, scale, gamma, multiplier_name, nu_type):
        alpha = gamma / 3.
        xi = 4 * alpha / (2 + alpha)
        poly_len = len('poly')
        if multiplier_name is None or multiplier_name == 'unity':
            multiplier = UnityMultiplier()
        elif multiplier_name == 'exp':
            multiplier = SmoothedExpMultiplier(scale/25.)
        elif multiplier_name.startswith('poly'):
            if len(multiplier_name) == poly_len:
                b = 1
            else:
                b = float(multiplier_name[poly_len:])
            multiplier = PolynomialMultiplier(1./scale, b)
        else:
            raise RuntimeError('invalid multiplier')
        sqrt_k = KSechMult(2. * scale, multiplier=multiplier)

        nu_scale = 2. * r * xi * scale
        if nu_type == 'mean':
            nu_sampler = lambda mean, X, n : sech_rvs(mean, nu_scale, n)
            nu_log_density = lambda x, mean, X: sech_logpdf(x, mean, nu_scale)
        elif nu_type == 'mixture':
            def nu_sampler(mean, X, n):
                grad_log_p = self.p.grad_log(X)
                wts = np.linalg.norm(grad_log_p, axis=1)
                wts /= np.sum(wts)
                inds = np.random.choice(wts.size, size=n, p=wts)
                return sech_rvs(X[inds], nu_scale, n)
            def nu_log_density (x, mean, X):
                lls = sech_logpdf(x[:,np.newaxis,:], X[np.newaxis,:,:], nu_scale)
                grad_log_p = self.p.grad_log(X)
                wts = np.linalg.norm(grad_log_p, axis=1)
                lls += np.log(wts)
                return misc.logsumexp(lls, axis=1) - np.log(np.sum(wts))
        elif nu_type.startswith('t-mean'):
            tmean_len = len('t-mean')
            if len(nu_type) == tmean_len:
                nu_df = 1.
            else:
                nu_df = float(nu_type[tmean_len:])
            nu_sampler = lambda mean, X, n: mean + stats.t.rvs(nu_df, scale=1./scale, size=(n, mean.size))
            nu_log_density = lambda x, mean, X: np.sum(stats.t.logpdf(x - mean, nu_df, scale=1./scale), axis=1)
        else:
            raise RuntimeError('invalid nu type')

        return sqrt_k, nu_sampler, nu_log_density

    def _feature_tensor(self, X, Z):
        # update data mean before using square root kernel
        self.f.center = self.data_mean
        return super(LrSechFastKSD, self)._feature_tensor(X, Z)

    def __str__(self):
        return '%s(scale=%f, gamma=%f)' % (type(self).__name__,
                                           self.scale, self.gamma)


class L2GaussFastKSD(FastKSD):
    def __init__(self, p, sigma2):
        sqrt_k, sampler, log_density = self._generate_sqrt_gaussian(sigma2)
        super(L2GaussFastKSD, self).__init__(2, sampler, log_density, sqrt_k, p)
        self.sigma2 = sigma2

    def _generate_sqrt_gaussian(self, sigma2):
        sqrt_k = KGauss2(sigma2 / 2.)

        nu_sigma2 = sigma2 / 2.
        def nu_sampler(mean, X, n):
            return np.sqrt(nu_sigma2) * (mean + np.random.randn(n, mean.size))
        def nu_log_density(x, mean, X):
            return -np.sum((x - mean)**2, axis=1) / (2 * nu_sigma2)

        return sqrt_k, nu_sampler, nu_log_density

    def __str__(self):
        return '%s(scale=%f, sigma2=%f)' % (type(self).__name__,
                                           self.sigma2)
