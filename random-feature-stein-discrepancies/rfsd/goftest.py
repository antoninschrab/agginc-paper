from __future__ import absolute_import, print_function

import numpy as np
import scipy

from kgof.goftest import H0Simulator, FSSD, GofTest, KernelSteinTest
from kgof.util import ContextTimer, NumpySeedContext

from .distributions import multivariate_iso_t_rvs, multivariate_iso_t_logpdf
from .util import SLArray



class FSSDH0SimCovDrawV(H0Simulator):
    """
    An asymptotic null distribution simulator for FSSD.  Simulate from the
    asymptotic null distribution given by the weighted sum of chi-squares. The
    eigenvalues (weights) are computed from the covarince matrix wrt. the
    sample drawn from p (the density to test against).

    - The UnnormalizedDensity p is required to implement get_datasource() method.
    """
    def __init__(self, n_draw=2000, n_simulate=3000, seed=10):
        """
        n_draw: number of samples to draw from the UnnormalizedDensity p
        """
        super(FSSDH0SimCovDrawV, self).__init__(n_simulate, seed)
        self.n_draw = n_draw

    def simulate(self, goft, dat, fea_tensor=None):
        """
        fea_tensor: n x d x J feature matrix

        This method does not use dat.
        """
        dat = None
        #assert isinstance(gof, FSSD)
        # p = an UnnormalizedDensity
        p = goft.p
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('DataSource associated with p must be available.')
        Xdraw = ds.sample(n=self.n_draw, seed=self.seed)
        _, fea_tensor = goft.compute_stat(Xdraw, return_feature_tensor=True)

        X = Xdraw.data()
        J = fea_tensor.shape[2]
        n = self.n_draw
        # n x d*J
        Tau = fea_tensor.reshape(n, -1)
        # Make sure it is a matrix i.e, np.cov returns a scalar when Tau is
        # 1d.
        #cov = np.cov(Tau.T) + np.zeros((1, 1))
        cov = Tau.T.dot(Tau)/n + np.zeros((1, 1))
        n_simulate = self.n_simulate

        seed = None if self.seed is None else self.seed + 20
        arr_nfssd, eigs = FSSD.list_simulate_spectral(cov, J, n_simulate,
                seed=seed)
        arr_nfssd += np.sum(eigs)
        #print('eigs:', eigs)
#         print('sum(eigs):', np.sum(eigs))
#         print('sum(eigs**2):', np.sum(eigs**2))
        return {'sim_stats': arr_nfsasd}


class FastKSDH0SimCovDrawV(H0Simulator):
    """
    An asymptotic null distribution simulator for L_p FastKSD.  Simulate from the
    asymptotic null distribution

        sum_i (sum_j |Z_ij|^p)^{2/p}

    where Z ~ N(0, Sigma) and Sigma is the covariance matrix estimated from a
    sample drawn from p (the density to test against).

    - The UnnormalizedDensity p is required to implement get_datasource() method.
    """
    def __init__(self, order=1, ordering='ij', n_draw=2000, n_simulate=3000, seed=10):
        """
        n_draw: number of samples to draw from the UnnormalizedDensity p
        """
        super(FastKSDH0SimCovDrawV, self).__init__(n_simulate, seed)
        self.order = order
        self.ordering = ordering
        self.n_draw = n_draw

    def simulate(self, goft, dat, fea_tensor=None):
        """
        fea_tensor: n x d x J feature matrix

        This method does not use dat.
        """
        dat = None
        #assert isinstance(gof, FSSD)
        # p = an UnnormalizedDensity
        p = goft.p
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('DataSource associated with p must be available.')
        Xdraw = ds.sample(n=self.n_draw, seed=self.seed)
        null_stat, fea_tensor = goft.compute_stat(Xdraw, return_feature_tensor=True)
        # print('null stat =', null_stat)

        X = Xdraw.data()
        n, d, J = fea_tensor.shape
        # n x d*J
        rescaling_of_Tau = np.max(fea_tensor)
        Tau = fea_tensor.reshape((n, -1)) / J**(1./self.order) / rescaling_of_Tau
        # print('FastKSDH0SimCovDrawV:')
        # print('    X shape:', X.shape)
        # print('    X:', X[:20].T)
        # print('    Tau:', Tau)
        # Make sure it is a matrix i.e, np.cov returns a scalar when Tau is
        # 1d.
        #cov = np.cov(Tau.T) + np.zeros((1, 1))
        cov = Tau.T.dot(Tau)/n + np.zeros((1, 1)) #1e-8*np.eye(d*J)
        cov_diag = np.diag(cov)
        rescaling_of_cov = np.max(cov_diag)
        valid_inds = cov_diag > 1e-15
        valid_cov = cov[np.ix_(valid_inds, valid_inds)] / rescaling_of_cov
        # print('    diag(cov)', cov_diag)
        # print('    diag(valid_cov)', np.diag(valid_cov))
        n_simulate = self.n_simulate

        block_size = max(20, int(10000.0/(d*J)))
        fksds = np.zeros(n_simulate)
        from_ind = 0
        zero_array = np.zeros(valid_cov.shape[0])
        while from_ind < n_simulate:
            to_draw = min(block_size, n_simulate-from_ind)
            # print(self.seed+from_ind+10)
            np.random.seed(self.seed+from_ind+10)
            # simulations = np.random.multivariate_normal(zero_array, cov, to_draw)
            partial_simulations = np.random.multivariate_normal(zero_array, valid_cov, to_draw)
            simulations = np.zeros((to_draw, d*J))
            simulations[:,valid_inds] = partial_simulations * rescaling_of_Tau * np.sqrt(rescaling_of_cov)
            simulations = simulations.reshape((to_draw,d,J))
            # if from_ind == 0:
            #     print(simulations[:2,:,3])

            # an array of length to_draw
            # if self.order == 1:
            #     sim_fksds_1 = np.sum(np.sum(np.abs(simulations), axis=2)**(2.), axis=1)
            simulations = np.abs(simulations)
            if self.order != 1:
                simulations **= self.order
            sim_fksds = np.sum(np.sum(simulations, axis=2)**(2./self.order), axis=1)
            # if self.order == 1:
            #     testing.assert_allclose(sim_fksds_1/sim_fksds, np.ones_like(sim_fksds_1))
            # store
            end_ind = from_ind+to_draw
            fksds[from_ind:end_ind] = sim_fksds
            from_ind = end_ind

        return {'sim_stats': fksds}


class RFDH0SimCovDrawV(H0Simulator):
    """
    An asymptotic null distribution simulator for an RFD.  Simulate from the
    asymptotic null distribution

        rfd.divergence()

    where Z ~ N(0, Sigma) and Sigma is the covariance matrix estimated from a
    sample drawn from p (the density to test against).

    - The UnnormalizedDensity p is required to implement get_datasource() method.
    """
    def __init__(self, n_draw=2000, n_simulate=3000, seed=None):
        """
        n_draw: number of samples to draw from the UnnormalizedDensity p
        """
        super(RFDH0SimCovDrawV, self).__init__(n_simulate, seed)
        self.n_draw = n_draw

    def simulate(self, goft, dat, fea_tensor=None, fea_weights=None):
        """
        fea_tensor: n x d x J feature matrix

        This method does not use dat.
        """
        sim_mode = 'svd'
        if sim_mode not in ['numpy', 'scipy', 'svd', 'tau_svd']:
            raise ValueError('invalid sim_mode:', sim_mode)
        dat = None
        fea_tensor = None
        fea_weights = None
        #assert isinstance(gof, FSSD)
        # p = an UnnormalizedDensity
        p = goft.p
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('DataSource associated with p must be available.')
        # sample data for covariance matrix estimation
        X = ds.sample(n=self.n_draw, seed=self.seed).data()
        null_stat, fea_tensor, fea_weights = goft.compute_stat(
            X, return_feature_info=True)
        # print('null stat =', null_stat)

        n, d, J = fea_tensor.shape
        fea_tensor *= fea_weights
        # n x d*J
        # rescaling_of_Tau = np.max(fea_tensor)

        if sim_mode == 'tau_svd':
            rescaling_of_Tau = fea_tensor.max() * np.sqrt(n)
            rescaled_Tau_array = fea_tensor.reshape((n, -1)) / rescaling_of_Tau
            rescaled_Tau = rescaled_Tau_array.toarray()
        else:
            Tau = fea_tensor.reshape((n, -1))
            cov = Tau.T.dot(Tau) / n # + 1e-8*np.eye(d*J)
            cov_diag = cov.diagonal()
            rescaling_of_cov = cov_diag.max()
            rescaled_cov = cov / rescaling_of_cov
            # print(rescaled_cov._log_a.min(), rescaled_cov._log_a.max())
            # exclude components with near-zero variance
            valid_inds = rescaled_cov.diagonal().toarray() > 1e-20
            res_cov_diag = rescaled_cov.diagonal().toarray()
            valid_rescaled_cov = rescaled_cov.toarray()[np.ix_(valid_inds, valid_inds)]
        n_simulate = self.n_simulate

        # can use this code to validate in order = (2,2) case
        if False and np.all(goft.rfd.order == (2,2)):
            print('using order (2,2) special case')
            rescaled_simulations, eigs = FSSD.list_simulate_spectral(
                valid_rescaled_cov, J, n_simulate, seed=self.seed+10)
            rescaled_simulations += np.sum(eigs)
            sim_stats = np.log(rescaled_simulations) + np.log(rescaling_of_cov.toarray())
        else:
            block_size = max(30, int(10000.0/(d*J)))
            sim_stats = np.zeros(n_simulate)
            from_ind = 0
            if sim_mode == 'svd':
                u, s2, vh = np.linalg.svd(valid_rescaled_cov)
                s = np.sqrt(s2[np.newaxis,:])
            elif sim_mode.endswith('py'):
                zero_array = np.zeros(valid_rescaled_cov.shape[0])
                if sim_mode == 'scipy':
                    mvn = scipy.stats.multivariate_normal(mean=zero_array,
                                                          cov=valid_rescaled_cov,
                                                          allow_singular=False)
            else:  # tau_svd
                u_tau, s_tau, vh_tau = np.linalg.svd(rescaled_Tau.T,
                                                     full_matrices=False)
                print(u_tau.shape, s_tau.shape, vh_tau.shape)
                su_tau = u_tau.dot(np.diag(s_tau)).T
            while from_ind < n_simulate:
                to_draw = n_simulate # min(block_size, n_simulate-from_ind)
                # print(self.seed+from_ind+10)
                np.random.seed(self.seed+from_ind+10)
                # components with zero variance are set to zero; others are sampled
                if sim_mode == 'tau_svd':
                    standard_normals = np.random.randn(to_draw, su_tau.shape[0])
                    rescaled_simulations = standard_normals.dot(su_tau)
                    simulations = SLArray.from_array(rescaled_simulations) * rescaling_of_Tau
                else:
                    if sim_mode == 'numpy':
                        rescaled_partial_simulations = np.random.multivariate_normal(
                            zero_array, valid_rescaled_cov, to_draw)
                    elif sim_mode == 'scipy':
                        rescaled_partial_simulations = mvn.rvs(to_draw)
                    elif sim_mode == 'svd':
                        standard_normals = np.random.randn(to_draw, s2.size)
                        rescaled_partial_simulations = (standard_normals * s).dot(vh)
                    rescaled_simulations = np.zeros((to_draw, d*J))
                    rescaled_simulations[:,valid_inds] = rescaled_partial_simulations
                    rescaled_simulations = rescaled_simulations.reshape((to_draw,d,J))
                    simulations = SLArray.from_array(rescaled_simulations) * rescaling_of_cov.sqrt()

                # if from_ind == 0:
                #     print(simulations[:2,:,3])

                # use simulated features to compute divergences
                # don't scale by n as this is the asymptotic distribution
                # of sqrt(n) RFD
                stats = goft.rfd.compute_divergence(simulations)
                if goft.rfd.log_scale():
                    stats *= 2
                else:
                    stats **= 2

                # for debugging purposes
                # print(simulations.shape)
                # sim_fksds = (simulations.abs().sum(axis=2)**(2.)).sum(axis=1)
                # print('  comparison')
                # print('    ', stats[:5])
                # print('    ', sim_fksds._log_a[:5])
                # testing.assert_allclose(stats/sim_fksds._log_a, np.ones_like(stats))
                # store
                end_ind = from_ind+to_draw
                sim_stats[from_ind:end_ind] = stats
                from_ind = end_ind
            #print('direct:  5%, 25%, 50%, 75% =', ', '.join([str(np.percentile(sim_stats, x)) for x in [5, 25, 50, 75]]))
        # print('bias    =', np.percentile(sim_stats, 50) - null_stat, ",") #, np.mean(sim_stats) - null_stat)
        # print('std dev =', np.std(sim_stats))
        #print(np.percentile(sim_stats, 50) - null_stat, ",") #np.std(sim_stats), ",")

        return {'sim_stats' : sim_stats, 'bias' : np.percentile(sim_stats, 50) - null_stat} # + null_stat - np.percentile(sim_stats, 50)}


class RFDH0DirectSim(H0Simulator):
    """
    Simulate directly from the RFD null distribution simulator for an RFD.

    - The UnnormalizedDensity p is required to implement get_datasource() method.
    """
    def __init__(self, n_simulate=1000, seed=None):
        super(RFDH0DirectSim, self).__init__(n_simulate, seed)

    def simulate(self, goft, dat, fea_tensor=None, fea_weights=None):
        """
        fea_tensor: n x d x J feature matrix

        This method does not use dat.
        """
        n, d, J = fea_tensor.shape
        dat = None
        fea_tensor = None
        fea_weights = None
        #assert isinstance(gof, FSSD)
        # p = an UnnormalizedDensity
        p = goft.p
        ds = p.get_datasource()
        if ds is None:
            raise ValueError('DataSource associated with p must be available.')
        # sample data for covariance matrix estimation
        # print(goft.rfd.__class__)
        sim_stats = np.zeros(self.n_simulate)
        for i in range(self.n_simulate):
            #if i+1 % 10 == 0:
            # print(i+1, end=' ')
            # sys.stdout.flush()
            # with Timer('draw X'):
            X = ds.sample(n=n, seed=self.seed+i).data()
            # with Timer('computer stat'):
            stat = goft.compute_stat(
                X, return_feature_info=False)
            sim_stats[i] = stat
        print()

        return {'sim_stats' : sim_stats}


class RFDGofTest(GofTest):
    def __init__(self, p, rfd, J=10, null_sim=None, alpha=.05):
        super(RFDGofTest, self).__init__(p, alpha)
        self.J = J
        self.null_sim = self._choose_null_sim(null_sim)
        self.rfd = rfd

    def _choose_null_sim(self, null_sim):
        if null_sim is not None:
            return null_sim
        return RFDH0SimCovDrawV()

    def perform_test(self, dat, return_simulated_stats=False):
        """
        dat: an instance of Data
        """
        with ContextTimer() as t:
            alpha = self.alpha
            null_sim = self.null_sim
            n_simulate = null_sim.n_simulate
            X = dat.data()
            n = X.shape[0]
            J = self.J

            stat, fea_tensor, fea_weights = self.compute_stat(
                X, return_feature_info=True)
            # if stat < SMALL_STAT_THRESHOLD and stat > 0:
            #     warnings.warn('statistic is very small (%e); there may be '
            #                  'numerical instability issues' % stat)
            sim_results = null_sim.simulate(self, dat, fea_tensor, fea_weights)
            sim_stats = sim_results['sim_stats']

            # approximate p-value with the permutations
            pvalue = np.mean(sim_stats > stat)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': stat,
                'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                'time_secs': t.secs, 'bias' : sim_results['bias']
                }
        if return_simulated_stats:
            results['sim_stats'] = sim_stats
        return results

    def compute_stat(self, X, return_feature_info=False):
        ret = self.rfd.divergence(
            X, J=self.J, reuse_features=True,
            return_feature_info=return_feature_info)
        base_stat = ret[0] if return_feature_info else ret
        if self.rfd.log_scale():
            stat = 2 * base_stat + np.log(X.shape[0])
            # print(np.log(X.shape[0]), X.shape)
        else:
            stat = X.shape[0] * base_stat**2
        if return_feature_info:
            return stat, ret[1], ret[2]
        else:
            return stat


class GeneralizedFSSD(FSSD):
    def __init__(self, p, k, V, order=2, null_sim=None, seed=None, alpha=.05):
        null_sim = self._choose_null_sim(null_sim, order)
        super(GeneralizedFSSD, self).__init__(p, k, V, alpha=alpha,
                                              null_sim=null_sim)
        self.order = order

    def _choose_null_sim(self, null_sim, order):
        if null_sim is not None:
            return null_sim
        elif order == 2:
            return FSSDH0SimCovDrawV(seed=None)
        else:
            return FastKSDH0SimCovDrawV(order=order, seed=None, n_draw=5000)

    def perform_test(self, dat, return_simulated_stats=False):
        """
        dat: an instance of Data
        """
        with ContextTimer() as t:
            alpha = self.alpha
            null_sim = self.null_sim
            n_simulate = null_sim.n_simulate
            X = dat.data()
            n, d = X.shape
            J = self.V.shape[0]

            nfssd, fea_tensor = self.compute_stat(dat, return_feature_tensor=True)
            # if nfssd < SMALL_STAT_THRESHOLD and nfssd > 0:
            #     warnings.warn('statistic is very small (%e); there may be '
            #                  'numerical instability issues' % nfssd)
            sim_results = null_sim.simulate(self, dat, fea_tensor)
            arr_nfssd = d * sim_results['sim_stats']

            # approximate p-value with the permutations
            pvalue = np.mean(arr_nfssd > nfssd)
            h0_rejected = pvalue < alpha

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': nfssd,
                'h0_rejected': h0_rejected, 'n_simulate': n_simulate,
                'time_secs': t.secs,
                }
        if return_simulated_stats:
            results['sim_stats'] = arr_nfssd
        return results


    def compute_stat(self, dat, return_feature_tensor=False):
        """
        The V-type statistic is
            sum_i (J^{-1} sum_j |V_ij|^p)^{2/p},
        where
            V_ij = N^{-1/2} sum_n T_i f(X_n, Z_j)
        """
        X = dat.data()
        n, d = X.shape

        # n x d x J
        Xi = self.feature_tensor(X)
        #stat = np.sum(np.sum(Xi, 0)**2) / Xi.shape[0]
        stats = np.abs(np.sum(Xi, axis=0)) / np.sqrt(Xi.shape[0])
        # print(stats)
        if self.order != 1:
            stats **= self.order
        stat = np.sum(np.mean(stats, axis=1)**(2./self.order))
        if return_feature_tensor:
            return stat, Xi
        else:
            return stat

    def feature_tensor(self, X):
        """
        Compute the feature tensor which is n x d x J.
        The feature tensor can be used to compute the statistic, and the
        covariance matrix for simulating from the null distribution.

        X: n x d data numpy array

        return an n x d x J numpy array
        """
        k = self.k
        J = self.V.shape[0]
        n, d = X.shape
        # n x d matrix of gradients
        grad_logp = self.p.grad_log(X)
        # n x J matrix
        K = k.eval(X, self.V)

        list_grads = np.array([np.reshape(k.gradX_y(X, v), (1, n, d)) for v in self.V])
        stack0 = np.concatenate(list_grads, axis=0)
        dKdV = np.transpose(stack0, (1, 2, 0))

        # n x d x J tensor
        grad_logp_K = np.einsum('ij,ik->ijk', grad_logp, K)
        Xi = grad_logp_K + dKdV
        # print(n, d, J)
        # print('K:', K)
        # print('grad_logp_K:', grad_logp_K)
        # print('dKdV:', dKdV)
        # print('Xi:', Xi)
        return Xi


class KernelSteinTest2(KernelSteinTest):
    """
    Goodness-of-fit test using kernelized Stein discrepancy test of
    Chwialkowski et al., 2016 and Liu et al., 2016 in ICML 2016.
    Mainly follow the details in Chwialkowski et al., 2016.
    The test statistic is n*V_n where V_n is a V-statistic.

    - This test runs in O(n^2 d^2) time.

    H0: the sample follows p
    H1: the sample does not follow p

    p is specified to the construct in the form of an UnnormalizedDensity.
    """

    def __init__(self, p, k, alpha=0.01, n_simulate=500, seed=None):
        """
        p: an instance of UnnormalizedDensity
        k: a KSTKernel object
        alpha: significance level
        n_simulate: The number of times to simulate from the null distribution.
            Must be a positive integer.
        """
        super(KernelSteinTest2, self).__init__(p, k, None, alpha,
                                               n_simulate, seed)

    def perform_test(self, dat, return_simulated_stats=False,
                     return_ustat_gram=False):
        """
        dat: a instance of Data
        """
        with ContextTimer() as t:
            alpha = self.alpha
            n_simulate = self.n_simulate
            X = dat.data()
            n = X.shape[0]

            _, H = self.compute_stat(dat, return_ustat_gram=True)
            test_stat = n*np.mean(H)
            # bootrapping
            sim_stats = np.zeros(n_simulate)
            ds = self.p.get_datasource()
            with NumpySeedContext(seed=self.seed):
                for i in range(n_simulate):
                    if (i+1) % 10 == 0:
                        print('iter', i+1)
                    dat_sim = ds.sample(n, seed=None)
                    _, H_sim = self.compute_stat(dat_sim, return_ustat_gram=True)
                    test_stat_sim = n*np.mean(H_sim)
                    sim_stats[i] = test_stat_sim

            # approximate p-value with the permutations
            pvalue = np.mean(sim_stats > test_stat)

        results = {'alpha': self.alpha, 'pvalue': pvalue, 'test_stat': test_stat,
                 'h0_rejected': pvalue < alpha, 'n_simulate': n_simulate,
                 'time_secs': t.secs,
                 }
        if return_simulated_stats:
            results['sim_stats'] = sim_stats
        if return_ustat_gram:
            results['H'] = H

        return results

# end KernelSteinTest2
