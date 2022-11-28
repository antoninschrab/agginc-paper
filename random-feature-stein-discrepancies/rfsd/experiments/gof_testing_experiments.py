from __future__ import absolute_import, print_function

from collections import defaultdict
import itertools
import sys

import autograd.numpy as np
from numpy.linalg.linalg import LinAlgError
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns

from kgof.data import DSLaplace, DSGaussBernRBM
from kgof.density import IsotropicNormal, GaussBernRBM
from kgof.goftest import GaussFSSD, KernelSteinTest
from kgof.kernel import KGauss, KIMQ
from kgof.util import fit_gaussian_draw

from rfsd.goftest import RFDH0SimCovDrawV, RFDGofTest
from rfsd.kernel import KGauss2
from rfsd.data import DSStudentsT
from rfsd.rfsd import L1IMQFastKSD, LrSechFastKSD, L2GaussFastKSD, RFFKSD
from rfsd.util import meddistance, binomial_interval
from rfsd.distributions import rff_cauchy_sampler
from . import config


class constant_dict():
    def __init__(self, const):
        self._const = const

    def __getitem__(self, k):
        return self._const

    def get(self, k, default=None):
        return self._const
# nominal levels when target level is 0.05
# depends on choice of gamma and dimension
L1_IMQ_NOMINAL_LEVELS = {
    .25 : {
        1: 0.032,
        3: 0.05,
        5: 0.048,
        7: 0.05,
        10: 0.025,
        15: 0.021,
        20: 0.011 },
    .5 : {
        1: 0.037,
        3: 0.037,
        5: 0.037,
        7: 0.037,
        10: 0.037,
        15: 0.03,
        20: 0.03 },
    1. : constant_dict(.05)
}

L2_SECHEXP_NOMINAL_LEVELS = {
    .25 : {
        1: 0.05,
        3: 0.042,
        5: 0.047,
        7: 0.034,
        10: 0.035,
        15: 0.032,
        20: 0.021 }
}

### Gaussian vs Laplace experiment ###

def construct_gauss_laplace_case(n, d):
    p = IsotropicNormal(np.zeros(d), 1)
    dat = DSLaplace(d, 0, 1/np.sqrt(2)).sample(n, seed=None)
    return p, dat


def run_gauss_laplace_goft_experiment(d=1, **kwargs):
    other_params = { 'd' : d}
    return run_goft_experiment(lambda n: construct_gauss_laplace_case(n, d),
                               other_params, **kwargs)


### Gaussian vs Gaussian experiment ###

def construct_gauss_case(n, d):
    p = IsotropicNormal(np.zeros(d), 1)
    dat = p.get_datasource().sample(n, seed=None)
    return p, dat

def run_gauss_goft_experiment(d=1, **kwargs):
    other_params = { 'd' : d}
    return run_goft_experiment(lambda n: construct_gauss_case(n, d),
                               other_params, **kwargs)

### Gaussian vs Student's T experiment ###

def construct_gauss_t_case(n, d, df):
    if df > 2:
        variance = df / (df - 2.0)
    else:
        variance = 1
    p = IsotropicNormal(np.zeros(d), variance)
    ds = DSStudentsT(d, df)
    dat = ds.sample(n, seed=None)
    return p, dat


def run_gauss_t_goft_experiment(d=1, df=10, **kwargs):
    other_params = { 'd' : d, 'df' : df}
    return run_goft_experiment(lambda n: construct_gauss_t_case(n, d, df),
                               other_params, **kwargs)


### RBM vs corrupted RBM experiment ###

def generate_rbm_case(n, dx, dh, sigmaPer):
    b = np.random.randn(dx)
    c = np.random.randn(dh)
    B = np.random.choice([-1,1],size=[dx, dh])
    Bcorrupted = B + sigmaPer * np.random.randn(dx, dh)

    p = GaussBernRBM(B, b, c)
    dat = DSGaussBernRBM(Bcorrupted, b, c).sample(n, seed=None)
    return p, dat

def run_rbm_fssd_experiment(dx=10, dh=5, sigmaPer=0.0, **kwargs):
    other_params = { 'dx' : dx, 'dh' : dh, sigmaPer : sigmaPer }
    return run_goft_experiment(lambda n: generate_rbm_case(n, dx, dh, sigmaPer),
                               other_params, **kwargs)


### shared experiment code ###

def single_gof_testing_round(test_names, case, gamma, test_alpha, J):
    # generate H0, data pair
    p, dat = case

    # We will use 20% of the data for parameter tuning, and 80% for testing.
    tr, te = dat.split_tr_te(tr_proportion=0.2, seed=None)
    X = dat.data()
    d = dat.dim()

    # Use median heuristics to choose hyperparameters
    med_l2 = meddistance(X, subsample=1000)
    med_l1 = meddistance(X, subsample=1000, metric='l1')
    scale = 1 / med_l1 / np.sqrt(np.pi/2)
    alpha = gamma / 3.
    xi = 4 * alpha / (2 + alpha)
    if d > 1:
         scale /= 2*xi
    sigma2 = med_l2**2
    c = 4 * med_l2
    c_rbm = 100 * med_l2

    # Generate feature locations for kernels
    Vgauss = fit_gaussian_draw(tr.data(), J, reg=1e-6,
                               seed=int(10*np.sum(tr.data()**2)))
    V0 = Vgauss
    if len(Vgauss.shape) == 0:
        Vgauss = np.array([[Vgauss]])
    if J == 1:
        Vgauss = Vgauss.T

    # Set up kernels and GoF tests
    kgauss = KGauss2(sigma2)

    if 'Gauss FSSD-opt' in test_names:
        # Use grid search to initialize the gwidth
        n_gwidth_cand = 5
        gwidth_factors = 2.0**np.linspace(-3, 3, n_gwidth_cand)
        kinit = KGauss(sigma2*2)
        list_gwidth = np.hstack( ( (sigma2)*gwidth_factors ) )
        besti, objs = GaussFSSD.grid_search_gwidth(p, tr, V0,
                                                       list_gwidth)
        gwidth = list_gwidth[besti]

        ops = {
            'reg': 1e-2,
            'max_iter': 40,
            'tol_fun': 1e-4,
            'disp': False,
            'locs_bounds_frac': 10.0,
            'gwidth_lb': 1e-1,
            'gwidth_ub': 1e4,
            }
        Vgauss_opt, gwidth_opt, info = GaussFSSD.optimize_locs_widths(p, tr,
                gwidth, V0, **ops)

    # L1 IMQ
    rfd_imq = L1IMQFastKSD(p, c=c, gamma=gamma, d=d, target_df=.5)
    rfd_imq_rbm = L1IMQFastKSD(p, c=c_rbm, gamma=gamma, d=d, target_df=2.6)
    # Lr Sech
    rfd_sech = LrSechFastKSD(p, scale=scale, gamma=gamma)
    # RFFs
    rff_gauss = RFFKSD(kgauss.rff_sampler(d), p)
    rff_cauchy = RFFKSD(rff_cauchy_sampler(med_l2, d), p)
    # KSDs
    kimq = KIMQ(c=1)

    sim_seed = hash(np.random.rand()) % np.iinfo(np.int32).max

    l1_rfd_null_sim = RFDH0SimCovDrawV(n_draw=5000+d*500, seed=sim_seed)
    l1_rbm_rfd_null_sim = RFDH0SimCovDrawV(n_draw=5000, seed=sim_seed)
    l2_rfd_null_sim = RFDH0SimCovDrawV(n_draw=5000, seed=sim_seed)

    l2_sech_test_alpha = test_alpha / (.9 + .1 * min(15, d))
    l1_imq_test_alpha = test_alpha / (.8 + .2 * d)
    if test_alpha == 0.05:
        if gamma in L2_SECHEXP_NOMINAL_LEVELS:
            l2_sech_test_alpha = L2_SECHEXP_NOMINAL_LEVELS[gamma].get(d, l2_sech_test_alpha)
        if gamma in L1_IMQ_NOMINAL_LEVELS:
            l1_imq_test_alpha = L1_IMQ_NOMINAL_LEVELS[gamma].get(d, l1_imq_test_alpha)
    if gamma == 1.:
        l1_imq_rbm_test_alpha = test_alpha
    elif gamma == .5:
        l1_imq_rbm_test_alpha = .94 * test_alpha
    else:
        l1_imq_rbm_test_alpha = .1 * test_alpha

    possible_tests = {    'L1 IMQ (alpha)' : RFDGofTest(p, rfd_imq, null_sim=l1_rfd_null_sim, alpha=test_alpha),
                          'L1 IMQ (RBM)' : RFDGofTest(p, rfd_imq_rbm, null_sim=l1_rbm_rfd_null_sim, alpha=l1_imq_rbm_test_alpha),
                          'L1 IMQ' : RFDGofTest(p, rfd_imq, null_sim=l1_rfd_null_sim, alpha=l1_imq_test_alpha),
                          'L2 SechExp' : RFDGofTest(p, rfd_sech, null_sim=l2_rfd_null_sim, alpha=l2_sech_test_alpha),
                          'L2 SechExp (alpha)' : RFDGofTest(p, rfd_sech, null_sim=l2_rfd_null_sim, alpha=test_alpha),
                          'Gauss FSSD-rand' : GaussFSSD(p, sigma2, Vgauss, alpha=test_alpha, seed=None),
                          'Gauss RFF' : RFDGofTest(p, rff_gauss, null_sim=l2_rfd_null_sim, alpha=test_alpha),
                          'Cauchy RFF' : RFDGofTest(p, rff_cauchy, null_sim=l2_rfd_null_sim, alpha=test_alpha),
                          'IMQ KSD' : KernelSteinTest(p, kimq, alpha=test_alpha, seed=None),
                          'Gauss KSD' : KernelSteinTest(p, kgauss, alpha=test_alpha, seed=None),
    }
    if 'Gauss FSSD-opt' in test_names:
        possible_tests['Gauss FSSD-opt'] = GaussFSSD(p, gwidth_opt, Vgauss_opt, alpha=test_alpha, n_simulate=2000, seed=None)
    tests = {}
    for tn in test_names:
        try:
            tests[tn] = possible_tests[tn]
        except KeyError:
            print('Invalid test name "{}"'.format(tn))
            raise

    results = {}
    for tn, t in tests.items():
        if 'opt' in tn:
            test_dat = te
        else:
            test_dat = dat
        results[tn] = t.perform_test(test_dat, return_simulated_stats=True)
    return results


def run_goft_experiment(generate_case, other_params={}, test_names=None,
                        gamma=.25, test_alpha=.05, n=1000,
                        J=5, rounds=25, verbose=True,
                        return_pvalues=False):
    if test_names is None:
        raise ValueError('test_names must be provided')
    test_rejects = defaultdict(int)

    p_values = defaultdict(list)
    biases = defaultdict(list)
    for i in range(rounds):
        if verbose:
            print(i+1, end=' ')
            sys.stdout.flush()
        case = generate_case(n)
        success = False
        while not success:
            try:
                results = single_gof_testing_round(test_names, case, gamma, test_alpha, J)
                success = True
            except LinAlgError as e:
                print('rerunning round because of error:', e)
        for tn, r in results.items():
            test_rejects[tn] += r['h0_rejected']
            if 'bias' in r:
                biases[tn].append(r['bias'])
            p_values[tn].append(r['pvalue'])

    for tn, blist in biases.items():
        print(tn, 'bias:', np.mean(blist))
    # Print out detailed results if there's only one round
    if rounds == 1:
        print(other_params)
        for tn, tr in results.items():
            print(tn, 'h0 rejected? ', tr['h0_rejected'])
        for tn, tr in results.items():
            print(tn, 'results:', tr)

    if verbose:
        print()
    params = {  'gamma' : gamma,
                'test_alpha' : test_alpha,
                'n' : n,
                'J' : J,
                'rounds' : rounds,
            }
    params.update(other_params)
    if return_pvalues:
        return test_rejects, p_values, params
    else:
        return test_rejects, params


def print_experiment_results(results):
    rejects, params = results
    rounds = params['rounds']
    for test_name, test_rejects in rejects.items():
        l, u = binomial_interval(.95, test_rejects, rounds)
        print('*', test_name, r'reject rate (95% confidence interval): ',
              float(test_rejects)/rounds,
              (round(l, 3), round(u, 3)))
        print(' ' * len(test_name), '  p-value for p = .05:',
              stats.binom_test(test_rejects, rounds, .05))


def plot_experiment_results(results, ys, ylabel, test_alpha,
                            ymax=1.05, plot_ci=None,
                            name_ordering=config.ORDERED_TEST_NAMES,
                            rounds=None, save=None, show=True,
                            legend_kwargs=None):
    color_dict = config.test_name_colors_dict()
    if name_ordering is None:
        name_ordering = results.keys()
    for show_legend in [True, False]:
        plt.figure()
        plt.clf()
        for test_name in name_ordering:
            if test_name in results:
                reject_rates = results[test_name]
                if test_name == 'L1 IMQ (RBM)':
                    test_name = 'L1 IMQ'
                c = color_dict[test_name]
                line, = plt.plot(ys, np.array(reject_rates), label=test_name, c=c)
                if plot_ci is not None:
                    l, u = zip(*[binomial_interval(.95, rounds*rr, rounds) for rr in reject_rates])
                    plt.fill_between(ys, l, u, alpha=0.25, facecolor=line.get_color())

        plt.plot((np.min(ys), np.max(ys)), (test_alpha, test_alpha), ':k')
        if show_legend:
            if legend_kwargs is None:
                plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
                           frameon=False)
            else:
                plt.legend(**legend_kwargs)
        if ymax is None:
            _, ymax = plt.ylim()
        plt.ylim(0, ymax)
        plt.xlabel(ylabel)
        plt.ylabel('null rejection rate')
        sns.despine()
        if save is not None:
            if show_legend:
                savefile = save + '-legend.png'
            else:
                savefile = save + '.png'
            plt.savefig(savefile, bbox_inches='tight')
        if show:
            plt.show()


def _format_list(l, f='%7s'):
    return ''.join([f % e for e in l])


def show_all_results(all_results, all_params, save=None, **kwargs):
    print('# of repetitions =', all_params['rounds'])
    for J, r in all_results.items():
        print('J =', J)
        table_str = 'test name \\ ' + all_params['variable_name'].split('$')[0]
        print('  %-30s' % table_str,
              _format_list(all_params['variable_values']))
        for tn, rrs in r.items():
            print('  %-30s' % tn, _format_list(rrs, '  %.3f'))
        if save is not None:
            savefile = '{}-J-{}'.format(save, J)
        else:
            savefile = None
        plot_experiment_results(r,
                                all_params['variable_values'],
                                all_params['variable_name'],
                                all_params['test_alpha'],
                                save=savefile,
                                **kwargs)


def run_goft_experiment_group(test_names, experiment, all_results=None,
                              all_params=None,
                              ds=[1,3,5,7,10], rounds=10, test_alpha=.05,
                              plot_results=True, Js=[5], **kwargs):
    if all_results is None:
        all_results = dict()
    if all_params is None:
        all_params = dict(variable_name='dimension $D$',
                          variable_values=ds,
                          rounds=rounds,
                          test_alpha=test_alpha)
        all_params.update(**kwargs)
        prev_rounds = 0
    else:
        # make sure all the parameters match
        assert all_params['test_alpha'] == test_alpha
        assert all_params['variable_values'] == ds
        assert all_params['variable_name'] == 'dimension $D$'
        for J in Js:
            assert J in all_results
        for k, v in kwargs.items():
            assert v == all_params[k]
        # update number of rounds
        prev_rounds = all_params['rounds']
        total_rounds = prev_rounds + rounds
        all_params['rounds'] = total_rounds
    print(all_params)

    for J in Js:
        if J not in all_results:
            all_results[J] = defaultdict(list)
        aggregated_results = all_results[J]
        for i, d in enumerate(ds):
            results = experiment(d=d, test_names=test_names, J=J, rounds=rounds,
                                 verbose=False, **kwargs)
            rejects, params = results
            if d == ds[0]:
                if J == Js[0]:
                    print('Shared Configuration:')
                    print('* rounds = {rounds}'.format(**params))
                    print('* ds =', ds)
                    print('* n = {n}, test_alpha = {test_alpha}'.format(**params))
                    print()
                print('### Experiment', i)
                print()
                print('Configuration:')
                print('* J = {J}'.format(**params))
                print()
            if not plot_results:
                print('d = {0}:'.format(d))
                print_experiment_results(results)
                print()
            else:
                print('d = {0} complete'.format(d))
            for tn, tr in rejects.items():
                if prev_rounds == 0:
                    aggregated_results[tn].append(float(tr) / rounds)
                else:
                    prev_tr = aggregated_results[tn][i] * prev_rounds
                    aggregated_results[tn][i] = (prev_tr + tr) / total_rounds

        all_results[J] = aggregated_results
        if plot_results:
            plot_experiment_results(aggregated_results, ds,
                                    'dimension $D$', params['test_alpha'])
    return all_results, all_params



def run_goft_rbm_experiment_group(test_names, experiment, all_results=None,
                                  all_params=None,
                                  sigmaPers=[0,.01,.02,.04,.06],
                                  rounds=10, test_alpha=.05,
                                  plot_results=True, Js=[5], **kwargs):
    if all_results is None:
        all_results = dict()
    if all_params is None:
        all_params = dict(variable_name='perturbation std. dev. $\sigma_{per}$',
                          variable_values=sigmaPers,
                          rounds=rounds,
                          test_alpha=test_alpha)
        all_params.update(**kwargs)
        prev_rounds = 0
    else:
        # make sure all the parameters match
        assert all_params['test_alpha'] == test_alpha
        assert all_params['variable_values'] == sigmaPers
        assert all_params['variable_name'] == 'perturbation std. dev. $\sigma_{per}$'
        for J in Js:
            assert J in all_results
        for k, v in kwargs.items():
            assert v == all_params[k]
        # update number of rounds
        prev_rounds = all_params['rounds']
        total_rounds = prev_rounds + rounds
        all_params['rounds'] = total_rounds
    print(all_params)

    for J in Js:
        if J not in all_results:
            all_results[J] = defaultdict(list)
        aggregated_results = all_results[J]
        for i, sigmaPer in enumerate(sigmaPers):
            results = experiment(sigmaPer=sigmaPer, test_names=test_names, J=J,
                                 rounds=rounds, verbose=False, **kwargs)
            rejects, params = results
            if sigmaPer == sigmaPers[0]:
                if J == Js[0]:
                    print('Shared Configuration:')
                    print('* rounds = {rounds}'.format(**params))
                    print('* sigmaPers =', sigmaPers)
                    print('* n = {n}, test_alpha = {test_alpha}'.format(**params))
                    print()
                print('### Experiment', i)
                print()
                print('Configuration:')
                print('* J = {J}'.format(**params))
                print()
            if not plot_results:
                print('sigmaPer = {0}:'.format(sigmaPer))
                print_experiment_results(results)
                print()
            else:
                print('sigmaPer = {0}:'.format(sigmaPer))
            for tn, tr in rejects.items():
                if prev_rounds == 0:
                    aggregated_results[tn].append(float(tr) / rounds)
                else:
                    prev_tr = aggregated_results[tn][i] * prev_rounds
                    aggregated_results[tn][i] = (prev_tr + tr) / total_rounds

        all_results[J] = aggregated_results
        if plot_results:
            plot_experiment_results(aggregated_results, sigmaPers,
                                    all_params['variable_name'],
                                    all_params['test_alpha'])
    return all_results, all_params
