from __future__ import absolute_import, print_function

import os
import argparse

import numpy as np

import seaborn as sns

import rfsd.experiments.efficiency_experiments as eff_exp
from rfsd.util import (Timer, store_objects, restore_object,
                       create_folder_if_not_exist, meddistance)
from rfsd.experiments import config
from rfsd import rfsd

from kgof import density, data


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output-dir', default='results')
    return parser.parse_args()


def main():
    sns.set_style('white')
    sns.set_context('notebook', font_scale=2, rc={'lines.linewidth': 3})
    args = parse_arguments()
    output_dir = args.output_dir

    create_folder_if_not_exist(output_dir)
    os.chdir(output_dir)
    print('changed working directory to', output_dir)

    n = 1000
    d = 10
    beta = -.5
    gammas = [1/2., 1/3., 1/4., 1/5.]
    Js = np.array([1,2,5,10,20,50])
    n_samps = 400
    J_large = 500

    dat = data.DSLaplace(d, 0, 1/np.sqrt(2)).sample(n, seed=None)
    X = dat.data()
    p = density.IsotropicNormal(np.zeros(d), 1)

    store_loc = 'efficiency-experiment-stored-data'

    try:
        IMQ_Y_hats = restore_object(store_loc, 'IMQ_Y_hats')
        IMQ_Y_bars = restore_object(store_loc, 'IMQ_Y_bars')
        print('reloaded existing IMQ data')
    except IOError:
        med_l2 = meddistance(X, subsample=1000, metric='l2')
        c = 4 * med_l2
        IMQ_Y_hats, IMQ_Y_bars = eff_exp.calculate_estimates(
            rfsd.L1IMQFastKSD, X, p, gammas, Js, J_large,
            number=n_samps, c=c, beta=beta, d=d,  ordering='ij',
            verbose=True, log_scale=True)
        store_objects(store_loc, IMQ_Y_hats=IMQ_Y_hats, IMQ_Y_bars=IMQ_Y_bars)
    IMQ_Y_stds, IMQ_Y_std_ratios, IMQ_probs = \
        eff_exp.calculate_efficiency_metrics(IMQ_Y_hats, IMQ_Y_bars, gammas, Js)

    try:
        sech_Y_hats = restore_object(store_loc, 'sech_Y_hats')
        sech_Y_bars = restore_object(store_loc, 'sech_Y_bars')
        print('reloaded existing Sech data')
    except IOError:
        med_l1 = meddistance(X, subsample=1000, metric='l1')
        scale = 1 / med_l1 / np.sqrt(np.pi/2)
        sech_Y_hats, sech_Y_bars = eff_exp.calculate_estimates(
            rfsd.LrSechFastKSD, X, p, gammas, Js, J_large,
            number=n_samps, scale=scale, verbose=True, log_scale=True)
        store_objects(store_loc, sech_Y_hats=sech_Y_hats, sech_Y_bars=sech_Y_bars)
    sech_Y_stds, sech_Y_std_ratios, sech_probs = \
        eff_exp.calculate_efficiency_metrics(sech_Y_hats, sech_Y_bars, gammas, Js)


    # plot the results
    eff_exp.plot_efficiency_metrics(IMQ_Y_stds, IMQ_Y_std_ratios,
                                    IMQ_Y_bars, IMQ_probs,
                                    gammas, Js, include_legend=True,
                                    show=False, save='L1-IMQ-efficiency')
    eff_exp.plot_efficiency_metrics(sech_Y_stds, sech_Y_std_ratios,
                                    sech_Y_bars, sech_probs,
                                    gammas, Js, include_legend=False,
                                    show=False, save='L2-SechExp-efficiency')
    eff_exp.plot_efficiency_comparison(gammas, None,
                                       'L2 SechExp', sech_Y_std_ratios,
                                       'L1 IMQ', IMQ_Y_std_ratios,
                                        show=False,
                                        save='sufficient-importance-sample-sizes')



if __name__ == '__main__':
    main()
