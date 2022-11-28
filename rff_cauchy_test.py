import sys
sys.path.append('random-feature-stein-discrepancies')
sys.path.append('wittawatj/kernel_gof')

from rfsd.goftest import RFDH0SimCovDrawV, RFDGofTest
from rfsd.rfsd import L1IMQFastKSD, RFFKSD
from rfsd.distributions import rff_cauchy_sampler
from kgof.density import GaussBernRBM
from kgof.data import DSLaplace, DSGaussBernRBM
import numpy as np
from rfsd.util import meddistance
from kgof.data import Data as kgof_Data

def rff_cauchy_test(X, p, seed, n_draw=5000):
    """
    parameters chosen from:
    https://bitbucket.org/jhhuggins/random-feature-stein-discrepancies/src/master/rfsd/experiments/gof_testing_experiments.py
    commit d77218b from 2018-12-21
    Comments mentioning the line from the above file are provided to justify the choices made.
    Those choices are coherent with the description given in Section 5 Experiments of:
    https://papers.nips.cc/paper/2018/file/0f840be9b8db4d3fbd5ba2ce59211f55-Paper.pdf
    """
    med_l2 = meddistance(X, subsample=1000)  # l146
    d = X.shape[1]
    rff_cauchy = RFFKSD(rff_cauchy_sampler(med_l2, d), p)  # 198

    sim_seed = seed
    l2_rfd_null_sim = RFDH0SimCovDrawV(n_draw=n_draw, seed=sim_seed)  # 206
    test_alpha = 0.05
    CauchyRFF = RFDGofTest(p, rff_cauchy, null_sim=l2_rfd_null_sim, alpha=test_alpha)  # 229

    data = kgof_Data(X)
    output = CauchyRFF.perform_test(data)  #l249
    return int(output['h0_rejected'])






