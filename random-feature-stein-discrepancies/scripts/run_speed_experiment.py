from __future__ import absolute_import, print_function

import os
import argparse
from collections import OrderedDict

from rfsd.util import (Timer, store_objects, restore_object,
                       create_folder_if_not_exist)
from rfsd.experiments import config
from rfsd import rfsd

from kgof import kernel, density, goftest
from kgof.util import fit_gaussian_draw

import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rounds', type=int, default=10)
    parser.add_argument('-o', '--output-dir', default='results')
    return parser.parse_args()


def main():
    sns.set_style('white')
    sns.set_context('notebook', font_scale=3, rc={'lines.linewidth': 3})
    args = parse_arguments()
    output_dir = args.output_dir

    create_folder_if_not_exist(output_dir)
    os.chdir(output_dir)
    print('changed working directory to', output_dir)

    ns = [100, 500, 1000, 2000, 5000]
    d = 10
    J = 10
    reps = args.rounds

    p = density.IsotropicNormal(np.zeros(d), 1)
    def make_divergence_call(k):
        return lambda dat: k.divergence(dat.data(), J=J)

    def fssd(dat):
        V = fit_gaussian_draw(dat.data(), J, seed=4, reg=1e-6)
        return goftest.GaussFSSD(p, 1, V).compute_stat(dat)

    metrics = OrderedDict([
        #('Gauss FSSD-rand', fssd),
        ('IMQ KSD', make_divergence_call(rfsd.KSD(kernel.KIMQ(), p))),
        ('L2 SechExp', make_divergence_call(rfsd.LrSechFastKSD(p))),
        ('L1 IMQ', make_divergence_call(rfsd.L1IMQFastKSD(p, d=d)))])

    store_loc = 'speed-experiment-stored-data-%d-rounds' % reps

    try:
        times = restore_object(store_loc, 'times')
        print('reloaded existing data')
    except IOError:
        times = OrderedDict([(k, np.zeros((len(ns), reps))) for k in metrics])
        for i, n in enumerate(ns):
            print('n =', n)
            for j in range(reps):
                dat = p.get_datasource().sample(n, seed=None)
                for kname, f in metrics.items():
                    with Timer() as t:
                        f(dat)
                    times[kname][i,j] = t.interval
        store_objects(store_loc, times=times)

    # plot the results
    base_fig_name = 'speed-experiment-%d-rounds' % reps
    color_dict = config.test_name_colors_dict()

    plt.figure()
    plt.clf()
    for kname, ktimes in times.items():
        ts = np.mean(ktimes, axis=1)
        plt.plot(ns, ts, label=kname, c=color_dict[kname])
    plt.xlabel('sample size $N$')
    plt.yscale('log')
    plt.ylabel('time (seconds)')
    sns.despine()
    plt.savefig(base_fig_name + '-no-legend.png', bbox_inches='tight')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., frameon=False)
    plt.savefig(base_fig_name + '.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
