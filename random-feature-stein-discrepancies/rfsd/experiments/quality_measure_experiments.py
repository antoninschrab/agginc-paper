from __future__ import absolute_import, print_function

import autograd.numpy as np
from autograd import elementwise_grad
from autograd.scipy.misc import logsumexp

import matplotlib.pyplot as plt
import seaborn as sns

from kgof.density import UDFromCallable
from kgof.kernel import KIMQ

from rfsd.rfsd import L1IMQFastKSD, LrSechFastKSD, KSD
from rfsd.util import create_folder_if_not_exist
from rfsd.inference import sgld, mh
from . import config


DEFAULT_EPS_LIST = [5e-2, 1e-2, 5e-3, 1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
DEFAULT_REPETITIONS = 20


def sgld_experiments(true_theta, log_prior, log_likelihood,
                     grad_log_prior, grad_log_likelihood, simulate,
                     N=100, batch_size=10, num_samples=2500,
                     eps_list=DEFAULT_EPS_LIST, repetitions=DEFAULT_REPETITIONS,
                     verbose=False, data_seed=None, sgld_seed=None):
    ## set up model ##
    if data_seed is not None:
        np.random.seed(data_seed)
    d = len(true_theta)
    X = simulate(true_theta, N)

    ## configure SGLD sampler ##
    num_sweeps = 2 * num_samples / (N / batch_size)

    ## run SGLD ##
    if sgld_seed is not None:
        np.random.seed(sgld_seed)
    samples = np.zeros((len(eps_list), repetitions, num_samples, d))
    theta0s = np.random.randn(repetitions, d)
    for i, eps in enumerate(eps_list):
        if verbose: print(i, eps)
        for j in range(repetitions):
            if verbose: print('  ', j+1, '/', repetitions)
            samples[i,j] = sgld(theta0s[j], X,
                                grad_log_prior, grad_log_likelihood,
                                eps, batch_size,
                                num_sweeps)[num_samples:]

    high_quality_samples = mala(true_theta, X, log_prior, log_likelihood,
                                grad_log_prior, grad_log_likelihood,
                                2*num_samples)

    return X, samples, high_quality_samples


def mala(theta0, X, log_prior, log_likelihood, grad_log_prior,
         grad_log_likelihood, num_samples):
    d = len(theta0)
    def prop_mean(theta, ell):
        grad = grad_log_likelihood(theta, X) - grad_log_prior(theta)
        return theta + .5 * np.exp(2 * ell) * grad
    def mala_q(theta, theta_new, ell):
        diff = prop_mean(theta, ell) - theta_new
        return -.5 * np.exp(-2 * ell) * np.sum(diff**2)
    def mala_samp_q(theta, ell):
        mean = prop_mean(theta, ell)
        return mean + np.exp(ell) * np.random.randn(d)
    p_full = lambda theta: log_likelihood(theta, X) + log_prior(theta)
    ell0 = np.log(.25)
    mala_samples, accept_rate = mh(theta0, p_full, mala_q, mala_samp_q,
                                   steps=2*num_samples, proposal_param=ell0,
                                   target_rate=.574)
    print('mala accept rate =', accept_rate)
    return np.array(mala_samples)


def standard_divergence_metrics(X, d, grad_log_prior, grad_log_likelihood,
                                gamma=.25, c=1., beta=-.5, scale=.5):
    grad_logp = lambda t: grad_log_normal_prior(t) + grad_log_mixture(t, X)
    p = UDFromCallable(d, fgrad_log=grad_logp)
    l1_imq_ksd = L1IMQFastKSD(p, c, beta, gamma, d, target_df=.5)
    l2_sech_ksd = LrSechFastKSD(p, scale=scale, gamma=gamma)
    imq_kernel = KIMQ(beta, c)
    imq_ksd = KSD(imq_kernel, p)
    metrics = { 'L1 IMQ' : l1_imq_ksd,
                'L2 SechExp' : l2_sech_ksd,
                'IMQ KSD': imq_ksd,
              }
    return metrics


def calculate_divergence_measures(samples, high_quality_samples, metrics,
                                  eps_list=DEFAULT_EPS_LIST,
                                  repetitions=DEFAULT_REPETITIONS, J=10,
                                  #ksd_divergence_values=dict(),
                                  verbose=False):
    ## compute quality metrics ##
    divergence_reps = 5
    divergence_values = { name : np.zeros((len(eps_list), repetitions, divergence_reps)) for name in metrics }
    for i, eps in enumerate(eps_list):
        if verbose: print(i, eps)
        for j in range(repetitions):
            if verbose: print('  ', j+1, '/', repetitions)
            for k in range(divergence_reps):
                for name, dvs in divergence_values.items():
                    if 'KSD' not in name:
                        dvs[i,j,k] = metrics[name].scaled_divergence(samples[i,j], J=J)
                    elif k == 0:
                        dvs[i,j,k] = metrics[name].scaled_divergence(samples[i,j,::5])
                    else:
                        dvs[i,j,k] = dvs[i,j,0]
    for name in divergence_values.keys():
        divergence_values[name] = divergence_values[name].reshape((len(eps_list), -1))
    hq_divergence_values = { }
    for name, dvs in divergence_values.items():
        if 'KSD' not in name:
            hq_divergence_values[name] = metrics[name].scaled_divergence(high_quality_samples, J=max(J, 250))
        else:
            hq_divergence_values[name] = metrics[name].scaled_divergence(high_quality_samples[::5])
    return divergence_values, hq_divergence_values


def plot_samples(X, samples, high_quality_samples, log_prior, log_likelihood,
                 eps_list=DEFAULT_EPS_LIST, show=True, save=None):
    delta = 0.025
    t1 = np.arange(-1, 2, delta)
    t2 = np.arange(-2.0, 2.0, delta)
    T1, T2 = np.meshgrid(t1, t2)
    T1r = T1.reshape((-1,))
    T2r = T2.reshape((-1,))
    eval_locs = np.stack((T1r, T2r)).T
    Zr = log_prior(eval_locs) + log_likelihood(eval_locs, X)
    Z = Zr.reshape(T1.shape)

    plt.figure()
    plt.clf()
    plt.plot(high_quality_samples[:,0], high_quality_samples[:,1], '.', alpha=.5)
    plt.contour(T1, T2, np.exp(Z))
    plt.title('high-quality samples')
    sns.despine()
    if save is not None:
        plt.savefig(save + '-high-quality.png', bbox_inches='tight')
    if show:
        plt.show()

    for i, eps in enumerate(eps_list):
        for j in range(samples.shape[1]):
            plt.clf()
            plt.plot(samples[i,j,:,0], samples[i,j,:,1], '.' ,alpha=.5)
            plt.contour(T1, T2, np.exp(Z))
            plt.title(r'$\epsilon$ = ' + str(eps))
            sns.despine()
            if save is not None:
                plt.savefig(save + '-eps-%s-%d.png' % (str(eps), j),
                            bbox_inches='tight')
            if show:
                plt.show()


def plot_sample_quality_results(divergence_values, hq_divergence_values,
                                eps_list=DEFAULT_EPS_LIST, J=None,
                                show=True, save=None):
    eps_line = [np.min(eps_list), np.max(eps_list)]
    #palette = sns.color_palette()
    color_dict = config.test_name_colors_dict()
    plt.figure()
    for i, (name, dvs) in enumerate(divergence_values.items()):
        c = color_dict[name] # palette[i]
        med_dvs = np.median(dvs, axis=1)
        rescaling = np.max(med_dvs)
        med_dvs /= rescaling
        hq_dv = hq_divergence_values[name] / rescaling
        optimal_index = np.argmin(med_dvs)
        plt.plot(eps_list, med_dvs, c=c, label=name.decode())
        #plt.plot(eps_line, [hq_dv, hq_dv], '--', c=c)
        plt.plot(eps_list[optimal_index], med_dvs[optimal_index], 'k*', ms=15)
        print('%-8s -' % name, eps_list[optimal_index])
    plt.xlabel(r'$\epsilon$')
    plt.ylabel('discrepancy measure')
    plt.xscale('log')
    plt.yscale('log')
    sns.despine()
    if J is not None:
        plt.title('$M$ = %d' % J)
    if save is not None:
        plt.savefig(save + '-no-legend.png', bbox_inches='tight')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False)
    if save is not None:
        plt.savefig(save + '.png', bbox_inches='tight')
    if show:
        plt.show()



## Toy mixture of Gaussians model from Welling and Teh (2011) ##

PRIOR_VARIANCES = np.array([10., 1.])

def log_normal_prior(theta):
    theta = np.atleast_2d(theta)
    return -np.sum(theta**2 / PRIOR_VARIANCES, axis=1) / 2

def grad_log_normal_prior(theta):
    return -theta / PRIOR_VARIANCES

def log_mixture(theta, X):
    theta = np.atleast_2d(theta)
    trans = np.array([[1,0],[1,1]])
    theta_trans = np.dot(trans, theta.T).T
    tiled_theta_trans = np.tile(theta_trans[:,:,np.newaxis], (1,1,X.shape[0]))
    tiled_X = np.tile(X.T, theta_trans.shape + (1,))
    squared_diffs = tiled_theta_trans**2 + tiled_X**2 - 2*np.einsum('ij,kl->ijk', theta_trans, X)
    return np.sum(logsumexp(-squared_diffs / 4., axis=1), axis=1)

grad_log_mixture = elementwise_grad(log_mixture)

def simulate_mixture_data(theta, N):
    trans = np.array([[1,0],[1,1]])
    theta_trans = np.dot(trans, theta)
    means = theta_trans[np.random.randint(2, size=N)]
    X = means + np.sqrt(2) * np.random.randn(N)
    return X[:,np.newaxis]


def run_mixture_posterior_experiment(separations=[1.], Js=[10], results=None,
                                     directory=None, verbose=False,
                                     skip_sample_plotting=False, show=True):
    sns.set_style('white')
    sns.set_context('notebook', font_scale=2, rc={'lines.linewidth': 3})
    if directory is not None:
        create_folder_if_not_exist(directory)
    if results is None:
        results = {}
    for i, separation in enumerate(separations):
        print('separation =', separation)
        true_theta = np.array([0., separation])
        seed_base = hash(separation)
        # run inference algorithms
        if separation not in results:
            results[separation] = sgld_experiments(
                true_theta, log_normal_prior, log_mixture,
                grad_log_normal_prior, grad_log_mixture,
                simulate_mixture_data, num_samples=1000,
                data_seed=5000+seed_base, sgld_seed=100+seed_base, # was: 10000, 100
                verbose=verbose)
        X, samples, high_quality_samples = results[separation]

        # calculate metrics
        np.random.seed(10+seed_base)
        metrics = standard_divergence_metrics(
            X, len(true_theta), grad_log_normal_prior, grad_log_mixture,
            scale = .5 / np.sqrt(np.pi/2), c=1)
        if not skip_sample_plotting:
            if directory is not None:
                save_loc = directory + '/separation-%s-samples' % str(separation)
            else:
                save_loc = None
            plot_samples(X, samples, high_quality_samples, log_normal_prior,
                         log_mixture, save=save_loc, show=show)
        for J in Js:
            b_metrics_key = (b'metrics', separation, J)
            if b_metrics_key not in results:
                print('computing metrics for key', b_metrics_key)
                results[b_metrics_key] = calculate_divergence_measures(
                    samples, high_quality_samples, metrics,
                    J=J, verbose=verbose)
            divergence_values, hq_divergence_values = results[b_metrics_key]
            if directory is not None:
                save_loc = directory + '/separation-%s-divergence-measures-J-%d' % (str(separation), J)
            plot_sample_quality_results(divergence_values, hq_divergence_values,
                                        J=J, save=save_loc, show=show)
    return results
