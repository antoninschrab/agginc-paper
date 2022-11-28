from __future__ import absolute_import, print_function

import argparse
from collections import defaultdict
import os
import sys

import numpy as np

import rfsd.experiments.gof_testing_experiments as goft_exp
from rfsd.util import (create_folder_if_not_exist,
                          pretty_file_string_from_dict,
                          nice_str, Timer,
                          store_objects, restore_object)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dims', metavar='dim', type=int, nargs='*',
                        default=[1,3,5,7,10,15,20])
    parser.add_argument('-r', '--rounds', type=int, default=0)
    parser.add_argument('-J', '--num-features', metavar='J', type=int,
                        default=10)
    parser.add_argument('-o', '--output-dir', default='results')
    parser.add_argument('-t', '--tests', nargs='*', metavar='test')
    parser.add_argument('-s', '--size', type=float, default=0)
    parser.add_argument('-g', '--gamma', type=float, default=0.25)
    parser.add_argument('-n', type=int, default=1000)
    parser.add_argument('--name', choices=['gauss', 'rbm'], default='gauss')
    return parser.parse_args()


def experiment_name(args):
    attributes = [('rounds', 'rounds'),
                  ('num_features', 'J'),
                  ('tests', 'tests'),
                  ('dims', 'dims')]
    if args.gamma != 0.25:
        attributes.append(('gamma', 'gamma'))
    attr_dict = { a[1] : nice_str(getattr(args, a[0])) for a in attributes }
    attr_str = pretty_file_string_from_dict(attr_dict)
    name = '-'.join(['estimate-nominal-thresholds', args.name, attr_str])
    return name


def estimate_thresholds(args, all_pvalues, all_thresholds):
    if all_pvalues is None:
        all_pvalues = defaultdict(list)
        for d in args.dims:
            print('d =', d)
            p_values = defaultdict(list)
            for i in range(args.rounds):
                if args.name == 'gauss':
                    case = goft_exp.construct_gauss_case(args.n, d)
                else:
                    case = goft_exp.generate_rbm_case(args.n, 50, 40, 0)
                results = goft_exp.single_gof_testing_round(
                    args.tests, case, args.gamma, 0, args.num_features)
                for tn, r in results.items():
                    p_values[tn].append(r['pvalue'])
            for tn, pv_list in p_values.items():
                all_pvalues[tn].append((d, pv_list))
    if all_thresholds is None:
        all_thresholds = dict()
    if args.size not in all_thresholds:
        thresholds = defaultdict(list)
        for tn, dim_pvals in all_pvalues.items():
            for d, pv_list in dim_pvals:
                threshold = np.percentile(pv_list, 100*args.size)
                thresholds[tn].append((d, threshold))
        all_thresholds[args.size] = thresholds
    return all_pvalues, all_thresholds


def main():
    args = parse_arguments()
    output_dir = args.output_dir

    create_folder_if_not_exist(output_dir)
    os.chdir(output_dir)
    print('changed working directory to', output_dir)
    if args.tests is None:
        if args.name == 'gauss':
            args.tests = ['L2 SechExp', 'L1 IMQ']
        else:  # args.name == 'rbm'
            args.tests = ['L2 SechExp', 'L1 IMQ (RBM)']
    if args.rounds <= 0:
        args.rounds = 200 if args.name == 'gauss' else 25
    if args.size <= 0:
        args.size = 0.05 if args.name == 'gauss' else 0.04
    if args.name == 'rbm':
        args.dims = [50]
    expt_name = experiment_name(args)
    store_loc = expt_name + '-stored-data'

    print('tests =', args.tests)
    print('size =', args.size)
    print('rounds =', args.rounds)
    print('dims =', args.dims)

    all_pvalues = None
    all_thresholds = None
    try:
        all_pvalues = restore_object(store_loc, 'all_pvalues')
        all_thresholds = restore_object(store_loc, 'all_thresholds')
        print('reloaded existing data')
    except IOError:
        pass
    with Timer('experiment'):
        all_pvalues, all_thresholds = estimate_thresholds(
            args, all_pvalues, all_thresholds)

    store_objects(store_loc, all_pvalues=all_pvalues,
                  all_thresholds=all_thresholds)

    for tn, dim_thresholds in all_thresholds[args.size].items():
        print(tn + ':')
        for d, threshold in dim_thresholds:
            print('  {:4>}: {:f}'.format(d, threshold))


if __name__ == '__main__':
    main()
