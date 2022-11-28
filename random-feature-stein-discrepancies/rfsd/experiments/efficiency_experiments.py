from __future__ import absolute_import, print_function

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from . import config


def calculate_estimates(cls, X, p, gammas, Js, J_large, number=100,
                        verbose=False, log_scale=False, **kwargs):
    Y_hats = np.zeros((len(gammas), len(Js), number))
    Y_bars = np.zeros(len(gammas))
    for i, gamma in enumerate(gammas):
        estimator = cls(p, gamma=gamma, **kwargs)
        Y_bars[i] = estimator.divergence(X, J=J_large)
        if verbose:
            print('gamma =', gamma)
            print('PhiSD = %e' % Y_bars[i])
        for j, J in enumerate(Js):
            if verbose:
                print('  M =', J)
            for n in range(number):
                Y_hats[i,j,n] = estimator.divergence(X, J=J)
    if log_scale:
        # rescale (since relative scaling is not important)
        return np.exp(Y_hats - Y_bars.reshape((-1,1,1))), np.exp(Y_bars - Y_bars)
    else:
        return Y_hats, Y_bars


PROB_INV_SCALE = 4

def calculate_efficiency_metrics(Y_hats, Y_bars, gammas, Js):
    probs = np.mean(Y_hats > Y_bars[:,np.newaxis,np.newaxis] / PROB_INV_SCALE,
                    axis=2)
    Y_stds = np.zeros(len(gammas))
    Y_std_ratios = np.zeros_like(Y_stds)
    for i, gamma in enumerate(gammas):
        errors = np.std(Y_hats[i], axis=1)
        Y_stds[i] = np.mean(errors * np.sqrt(Js))
        Y_std_ratios[i] = Y_stds[i] / Y_bars[i]
    return Y_stds, Y_std_ratios, probs


def plot_efficiency_comparison(gammas, n, *args, **kwargs):
    if len(args) % 2 == 1:
        raise RuntimeError('expected even number of additional arguments')
    print('choices of sample size necessary for ratio of std(Y_hat)/Y_bar to be 1/2')
    show = kwargs.get('show', True)
    save = kwargs.get('save', 'sufficient-importance-sample-sizes')
    gammas = np.asarray(gammas)
    num_samples = lambda r: 4*r**2
    #palette = sns.color_palette()
    color_dict = config.test_name_colors_dict()
    plt.figure()
    for i in range(int(len(args)/2)):
        name = args[2*i]
        c = color_dict[name]
        Y_std_ratios = args[2*i+1]
        y = num_samples(Y_std_ratios)
        if n is not None:
            err = Y_std_ratios * np.sqrt(2 / n)
            ylower = num_samples(Y_std_ratios - 2*err)
            yupper = num_samples(Y_std_ratios + 2*err)
            yerr1 = np.ceil(np.array([y - ylower, yupper - y]))
            y = np.ceil(y)
            yerr2 = np.array([y - np.floor(ylower), np.ceil(yupper) - y])
            yerr = np.maximum(yerr1, yerr2)
            plt.errorbar(gammas + .002*(i-(len(args)/2)/2.), y, yerr=yerr,
                         label=name, c=c)
        else:
            y = np.ceil(y)
            plt.plot(gammas, y, label=name, c=c)

    plt.yscale('log')
    ymin, ymax = plt.ylim()
    plt.ylim([1, int(ymax+1)])
    plt.xlabel(r'$\gamma$')
    plt.ylabel(r'importance sample size $M$')
    plt.legend(frameon=False)
    sns.despine()
    if save is not None:
        plt.savefig(save + '.png', bbox_inches='tight')
    if show:
        plt.show()


def plot_efficiency_metrics(Y_stds, Y_std_ratios, Y_bars, probs, gammas, Js,
                            include_legend=True, show=True, save=None):
    palette = sns.color_palette()
    plt.figure()
    for i, gamma in enumerate(gammas):
        std_ratios = Y_std_ratios[i] / np.sqrt(Js)
        plt.plot(Js, std_ratios, label=r'$\gamma = %.3f$' % gamma, c=palette[i])
    plt.yscale('log')
    ymin, ymax = plt.ylim()
    plt.ylim((min(1, ymin), ymax))
    plt.xlabel(r'importance sample size $M$')
    plt.ylabel(r'$\frac{\mathrm{stdev}(\mathrm{R\Phi SD})}{\Phi\mathrm{SD}}$')
    if include_legend:
        plt.legend(frameon=False)
    sns.despine()
    if show:
        plt.show()

    plt.clf()
    plt.plot(Js, probs.T)
    plt.xlabel(r'importance sample size $M$')
    plt.ylabel(r'$\mathrm{\mathbb{P}}[\mathrm{R\Phi SD} > \Phi\mathrm{SD}/%d]$' % PROB_INV_SCALE)
    if include_legend:
        plt.legend([r'$\gamma = %.3f$' % gamma for gamma in gammas],
                   frameon=False)
    #plt.xscale('log')
    sns.despine()
    if save is not None:
        plt.savefig(save + '.png', bbox_inches='tight')
    if show:
        plt.show()
