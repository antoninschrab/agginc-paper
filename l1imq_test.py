import sys
sys.path.append('random-feature-stein-discrepancies')
sys.path.append('wittawatj/kernel_gof')

from rfsd.goftest import RFDH0SimCovDrawV, RFDGofTest
from rfsd.rfsd import L1IMQFastKSD
from kgof.density import GaussBernRBM
from kgof.data import DSLaplace, DSGaussBernRBM
import numpy as np
from rfsd.util import meddistance
from kgof.data import Data as kgof_Data

def L1_IMQ_test(X, p, seed, corrected_level=True, n_draw=5000):
    """
    parameters chosen from:
    https://bitbucket.org/jhhuggins/random-feature-stein-discrepancies/src/master/rfsd/experiments/gof_testing_experiments.py
    commit d77218b from 2018-12-21
    Comments mentioning the line from the above file are provided to justify the choices made.
    Those choices are coherent with the description given in Section 5 Experiments of:
    https://papers.nips.cc/paper/2018/file/0f840be9b8db4d3fbd5ba2ce59211f55-Paper.pdf
    with the two exceptions:
    c_rbm = 10 * med_l2 in the paper while c_rbm = 100 * med_l2 is used in implementation
    target_df is set to 2.6 in implementation rather than 2.5 in the paper
    We follow the parameters used in their implementation since those are the ones used to reproduce the experiments.
    """
    gamma = 0.25  # l254
    med_l2 = meddistance(X, subsample=1000)  # l146
    c_rbm = 100 * med_l2  # l155
    d = X.shape[1]
    rfd_imq_rbm = L1IMQFastKSD(p, c=c_rbm, gamma=gamma, d=d, target_df=2.6)  # l193

    sim_seed = seed
    l1_rbm_rfd_null_sim = RFDH0SimCovDrawV(n_draw=n_draw, seed=sim_seed)  # l205
    test_alpha = 0.05
    if corrected_level:
        l1_imq_rbm_test_alpha = .1 * test_alpha # l220 for gamma = 0.25
    else:
        l1_imq_rbm_test_alpha = test_alpha
    L1IMQ = RFDGofTest(p, rfd_imq_rbm, null_sim=l1_rbm_rfd_null_sim, alpha=l1_imq_rbm_test_alpha)  # l223

    data = kgof_Data(X)
    output = L1IMQ.perform_test(data)  #l249
    return int(output['h0_rejected'])
 
