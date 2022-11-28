from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

from kgof.util import dist2_matrix
from kgof.kernel import KIMQ, KGauss, DifferentiableKernel, KSTKernel, Kernel

import autograd
import autograd.numpy as np


class LogScaleKernel(Kernel):
    __metaclass__ = ABCMeta

    @abstractmethod
    def log_eval(self, X, Y):
        raise NotImplementedError()

    def log_gradX_y(self, X, y):
        # we use the fact that log |grad f| = log |f| + log |grad log f|
        yrow = np.reshape(y, (1, -1))
        logk = lambda X: self.log_eval(X, yrow)
        grad_logk = autograd.elementwise_grad(logk)(X)
        KXy = self.eval(X, yrow)
        log_grad = np.log(np.abs(grad_logk)) + np.log(KXy)
        signs = np.sign(grad_logk)
        assert log_grad.shape[0] == X.shape[0]
        assert log_grad.shape[1] == X.shape[1]
        return log_grad, signs


class RFFKernel(Kernel):
    __metaclass__ = ABCMeta

    @abstractmethod
    def rff_sampler(self, d):
        raise NotImplementedError()


class KIMQ2(KIMQ, LogScaleKernel):
    def log_eval(self, X, Y):
        b = self.b
        c = self.c
        D2 = dist2_matrix(X, Y)
        logK = b * np.log(c**2 + D2)
        return logK


class KGauss2(KGauss, LogScaleKernel, RFFKernel):
    def log_eval(self, X, Y):
        sumx2 = np.reshape(np.sum(X**2, 1), (-1, 1))
        sumy2 = np.reshape(np.sum(Y**2, 1), (1, -1))
        D2 = sumx2 - 2*np.dot(X, Y.T) + sumy2
        logK = -D2 / (2.0 * self.sigma2)
        return logK

    def rff_sampler(self, d):
        sigma = np.sqrt(self.sigma2)
        return lambda n: np.random.randn(n,d) / sigma


class KernelMultiplier(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def log_eval(self, X):
        """Evaluate the log at the set of given locations

        Parameters
        ----------
        X : n x d numpy array

        Return
        ------
        vals : an n-dimensional numpy array
        """
        raise NotImplementedError()

    @abstractmethod
    def eval(self, X):
        """Evaluate at the set of given locations

        Parameters
        ----------
        X : n x d numpy array

        Return
        ------
        vals : an n-dimensional numpy array
        """
        raise NotImplementedError()

    @abstractmethod
    def gradLog(self, X, dim):
        """Evaluate the gradient of the log multiplier wrt a single dimension the set of given locations

        Parameters
        ----------
        X : n x d numpy array
        dim : int

        Return
        ------
        grads : an n-dimensional numpy array
        """
        raise NotImplementedError()


class UnityMultiplier(KernelMultiplier):
    def log_eval(self, X):
        return np.zeros(1)

    def eval(self, X):
        return np.ones(1)

    def gradLog(self, X, dim):
        return np.zeros(1)


class SmoothedExpMultiplier(KernelMultiplier):
    def __init__(self, scale):
        assert scale > 0, 'scale must be > 0. Is %s' % str(scale)
        self.scale = scale

    def log_eval(self, X):
        smoothedX = self.scale * np.sqrt(1 + X**2)
        return np.sum(smoothedX, axis=1)

    def eval(self, X):
        return np.exp(self.log_eval(X))

    def gradLog(self, X, dim):
        return self.pair_gradLog(X[:,[dim]])

    def pair_gradLog(self, X):
        return self.scale * X / np.sqrt(1 + X**2)


class PolynomialMultiplier(KernelMultiplier):
    """
    prod_i (a^2 + |x_i|^2)^{b/2}
    """
    def __init__(self, a, b):
        assert a > 0 and b > 0, 'must have a, b > 0.'
        self.a2 = a**2
        self.b = b

    def log_eval(self, X):
        logaxib = .5 * self.b * np.log(self.a2 + X**2)
        return np.sum(logaxib, axis=1)

    def eval(self, X):
        return np.exp(self.log_eval(X))

    def gradLog(self, X, dim):
        return self.pair_gradLog(X[:,[dim]])

    def pair_gradLog(self, X):
        return self.b * X / (self.a2 + X**2)


class KSechMult(LogScaleKernel, DifferentiableKernel, KSTKernel):

    def __init__(self, scale, multiplier, center=None):
        assert scale > 0, 'scale must be > 0. Is %s' % str(scale)
        self.scale = np.sqrt(np.pi/2) * scale
        self.multiplier = multiplier
        self.center = center

    def scale_parameter(self):
        return self.scale / np.sqrt(np.pi/2)

    def _recenter(self, X):
        if self.center is None:
            return X
        else:
            return X - self.center

    def log_eval(self, X, Y):
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'

        A_X = self.multiplier.log_eval(self._recenter(X))
        A_Y = self.multiplier.log_eval(self._recenter(Y))
        logK = A_X[:,np.newaxis] + A_Y[np.newaxis,:]
        Xs = self.scale * X
        Ys = self.scale * Y
        for i in range(d1):
            scaledDiff = Xs[:,[i]] - Ys[:,[i]].T
            logK = logK - np.logaddexp(scaledDiff, -scaledDiff)
        logK += d1 * np.log(2)
        return logK

    def eval(self, X, Y):
        """
        Evaluate the hyberbolic secant kernel with exponential rescaling on the two 2d numpy arrays.

        Parameters
        ----------
        X : n1 x d numpy array
        Y : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'

        A_X = self.multiplier.eval(self._recenter(X))
        A_Y = self.multiplier.eval(self._recenter(Y))
        K = np.outer(A_X, A_Y)
        Xs = self.scale * X
        Ys = self.scale * Y
        for i in range(d1):
            scaledDiff = Xs[:,[i]] - Ys[:,[i]].T
            K = K / np.cosh(scaledDiff)
        return K

    def gradX_Y(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        K = self.eval(X, Y)
        tanh = self.scale * np.tanh(self.scale * (X[:,[dim]] - Y[:,[dim]].T))
        gradLogMult = self.multiplier.gradLog(self._recenter(X), dim)
        G = K * (gradLogMult - tanh)
        return G

    def pair_gradX_Y(self, X, Y):
        """
        Compute the gradient with respect to X in k(X, Y), evaluated at the
        specified X and Y.

        X: n x d
        Y: n x d

        Return a numpy array of size n x d
        """
        Kvec = self.pair_eval(X, Y)
        tanh = self.scale * np.tanh(self.scale * (X - Y))
        gradLogMult = self.multiplier.pair_gradLog(self._recenter(X))
        G = Kvec[:, np.newaxis] * (gradLogMult - tanh)
        return G

    def gradY_X(self, X, Y, dim):
        """
        Compute the gradient with respect to the dimension dim of X in k(X, Y).

        X: nx x d
        Y: ny x d

        Return a numpy array of size nx x ny.
        """
        return self.gradX_Y(Y, X, dim).T

    def pair_gradY_X(self, X, Y):
        """
        Compute the gradient with respect to Y in k(X, Y), evaluated at the
        specified X and Y.

        X: n x d
        Y: n x d

        Return a numpy array of size n x d
        """
        return self.pair_gradX_Y(Y, X)

    def gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: nx x d numpy array.
        Y: ny x d numpy array.

        Return a nx x ny numpy array of the derivatives.
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert d1==d2, 'Dimensions of the two inputs must be the same'
        d = d1
        scale = self.scale
        # d_{x_i} d_{y_i} k1 k2 = k1 k2 [ d_{x_i} d_{y_i} log k_1 + d_{x_i} d_{y_i} log k_2
        #   + (d_{x_i} log k_1 + d_{x_i} log k_2)(d_{y_i} log k_1 + d_{y_i} log k_2)]
        # First evaluate the kernel at all pairs
        K = self.eval(X, Y)
        # Then, for each dimension i, construct coefficients based on i-th partial
        # derivatives of the kernel
        H = np.zeros((n1, n2))
        for i in range(d):
            # Introduce second derivatives of log base kernel: scale^2 sech(scale*(x-y))^2
            scaledDiff = scale * (X[:,[i]] - Y[:,[i]].T)
            H = H + scale**2 / np.cosh(scaledDiff)**2
            # Note that second derivatives of log multiplier = 0
            # Introduce grad log terms
            # gradLogMult matrices have size n x 1: derivatives of log multiplier
            gradLogMultX = self.multiplier.gradLog(self._recenter(X), i)
            gradLogMultY = self.multiplier.gradLog(self._recenter(Y), i)
            # gradLogBaseKernelX has size n x n: derivatives of log base kernel, -scale tanh(scale*(x-y))
            # Note that gradLogBaseKernelY = -gradLogBaseKernelX
            gradLogBaseKernelX = -scale * np.tanh(scaledDiff)
            H = H + (gradLogMultX + gradLogBaseKernelX) * (gradLogMultY.T - gradLogBaseKernelX)
        G = H * K
        return G

    def pair_gradXY_sum(self, X, Y):
        """
        Compute \sum_{i=1}^d \frac{\partial^2 k(X, Y)}{\partial x_i \partial y_i}
        evaluated at each x_i in X, and y_i in Y.

        X: n x d numpy array.
        Y: n x d numpy array.

        Return a one-dimensional length-n numpy array of the derivatives.
        """
        raise NotImplementedError()

    def gradX_y(self, X, y):
        yrow = np.reshape(y, (1, -1))
        f = lambda X: self.eval(X, yrow)
        g = autograd.elementwise_grad(f)
        G = g(X)
        assert G.shape[0] == X.shape[0]
        assert G.shape[1] == X.shape[1]
        return G

    def pair_eval(self, X, Y):
        """
        Evaluate k(x1, y1), k(x2, y2), ...

        Parameters
        ----------
        X, Y : n x d numpy array

        Return
        -------
        a numpy array with length n
        """
        (n1, d1) = X.shape
        (n2, d2) = Y.shape
        assert n1==n2, 'Two inputs must have the same number of instances'
        assert d1==d2, 'Two inputs must have the same dimension'
        diff = X - Y
        cosh = np.prod(np.cosh(self.scale * diff), axis=1)
        A_X = self.multiplier.eval(self._recenter(X))
        A_Y = self.multiplier.eval(self._recenter(Y))
        Kvec = A_X * A_Y / cosh
        return Kvec

    def __str__(self):
        return "KSechMult(%.3f)" % self.scale_parameter()
