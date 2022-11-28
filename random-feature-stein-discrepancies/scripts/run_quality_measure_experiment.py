from __future__ import absolute_import, print_function

import os
import argparse

import rfsd.experiments.quality_measure_experiments as q_exp
from rfsd.util import store_objects, restore_object, create_folder_if_not_exist

import seaborn as sns


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-sample-plotting', action='store_true')
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


    store_loc = 'sample-quality-experiment-stored-data'

    restored = False
    try:
        results = restore_object(store_loc, 'results')
        restored = True
        print('reloaded existing results')
    except IOError:
        results = dict()

    separations = [1.]
    Js = [10, 25, 50, 75]
    results = q_exp.run_mixture_posterior_experiment(
        separations=separations, verbose=False, results=results,
        skip_sample_plotting=args.skip_sample_plotting, directory='.',
        Js=Js, show=False)

    if not restored:
        store_objects(store_loc, results=results)


if __name__ == '__main__':
    main()
