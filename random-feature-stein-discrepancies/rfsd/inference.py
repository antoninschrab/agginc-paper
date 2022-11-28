from __future__ import absolute_import, print_function

import time

import numpy as np
import numpy.random as npr


def sgld(theta0, X, grad_logprior, grad_loglik, eps, batch_size, num_sweeps):
    """
    Run stochastic gradient Langevin dynamics

    grad_loglik : function
        Takes parameter value and data as arguments
    """
    d = theta0.shape[0]
    N = X.shape[0]
    assert batch_size <= N and batch_size > 0, 'batch size should be positive and at most the dataset size'
    rescaling_factor = float(N) / batch_size
    num_batches = int(rescaling_factor)
    thetas = np.zeros((num_sweeps * num_batches, d))
    theta_t = theta0.copy()
    t = 0
    for i in range(num_sweeps):
        perm = np.random.permutation(N)
        for b in range(num_batches):
            if num_batches == 1:
                X_subset = X
            else:
                X_subset = X[perm[b*batch_size:(b+1)*batch_size],:]
                assert X_subset.shape[0] == batch_size
            stochastic_grad = grad_logprior(theta_t) + rescaling_factor * grad_loglik(theta_t, X_subset)
            noise = np.sqrt(eps) * np.random.randn(d)
            theta_t += .5 * eps * stochastic_grad.squeeze() + noise
            thetas[t,:] = theta_t
            t += 1

    return thetas


def _ensure_positive_int(val, name):
    if not isinstance(val, int) or val <= 0:
        raise ValueError("'%s' must be a positive integer")
    return True


def _ensure_callable(val, name):
    if not callable(val):
        raise ValueError("'%s' must be a callable")
    return True


def _adapt_param(value, i, log_accept_prob, target_rate, const=3):
    """
    Adapt the value of a parameter.
    """
    new_val = value + const*(np.exp(log_accept_prob) - target_rate)/np.sqrt(i+1)
    # new_val = max(min_val, min(max_val, new_val))
    return new_val


def mh(x0, p, q, sample_q, steps=1, warmup=None, thin=1,
       proposal_param=None, target_rate=0.234, time_iters=None,
       adapt_until=None):
    """
    (Adaptive) Metropolis Hastings sampling.

    Parameters
    ----------
    x0 : object
        The initial state.

    p : function
        Accepts one argument `x` and outputs the log probability density of the
        target distribution at `x`.

    q : function or None
        Accepts two arguments, `x` and `xf`. Outputs the log proposal density
        of going from `x` to `xf`. None indicates the proposal is symmetric,
        so there is no need to calculate the proposal probability when
        deciding whether to accept the move to `xf`.

    sample_q : function
        Accepts one argument `x` and proposes `xf` given `x`.

    steps : int, optional
        The number of MH steps to take. Default is 1.

    warmup : int, optional
        The number of warmup (aka burnin) iterations. Default is ``steps/2``.

    thin : int, optional
        Period for saving samples. Default is 1.

    proposal_param : numeric, optional
        If provided then use adaptive MH targeting an accept rate of
        `target_rate`. In this case `sample_q` and `q` should both accept
        `proposal_param` as an additional final argument. Default is None.

    target_rate : float, optional
        Default is 0.234.

    time_iters : array-like, shape=(num_times,), optional
        If provided, then record times on iterations ``warmup + time_iters``.

    adapt_until : int, optional
        If provided, adapt for this many iterations. Otherwise adapt during
        warmup.

    Returns
    -------
    samples : array with length ``(steps - warmup) / thin``

    accept_rate : float
        Calculated from non-warmup iterations.

    times : array with length num_times
        Only returned if time_iters is not None.
    """
    # Validate parameters
    _ensure_callable(p, 'p')
    if q is not None:
        _ensure_callable(q, 'q')
    _ensure_callable(sample_q, 'sample_q')
    _ensure_positive_int(steps, 'steps')
    _ensure_positive_int(thin, 'thin')
    if warmup is None:
        warmup = steps / 2
    else:
        _ensure_positive_int(warmup + 1, 'warmup')
        if warmup >= steps:
            raise ValueError("Number of warmup iterations is %d, which is "
                             "greater than the total number of steps, %d" %
                             (warmup, steps))
    if adapt_until is None:
        adapt_until = warmup
    else:
        _ensure_positive_int(adapt_until + 1, 'adapt_until')
    if target_rate is None:
        target_rate = 0.234
    if time_iters is not None:
        assert time_iters.dtype == np.int
        ti_index = 0
        times = []
        start_time = time.clock()
    # Run (adaptive) MH algorithm
    accepts = 0.0
    xs = []
    x = x0
    for step in range(steps):
        # Make a proposal
        p0 = p(x)
        if proposal_param is None:
            xf = sample_q(x)
        else:
            xf = sample_q(x, proposal_param)
        pf = p(xf)

        # Compute acceptance ratio and accept or reject
        odds = pf - p0
        if q is not None:
            if proposal_param is None:
                qf, qr = q(x, xf), q(xf, x)
            else:
                qf, qr = q(x, xf, proposal_param), q(xf, x, proposal_param)
            odds += qr - qf
        if proposal_param is not None and step < adapt_until:
                proposal_param = _adapt_param(proposal_param, step,
                                              min(0, odds), target_rate)
        if np.log(npr.rand()) < odds:
            x = xf
            if step >= warmup:
                accepts += 1

        if step >= warmup and (step - warmup) % thin == 0:
            xs.append(x)

        if time_iters is not None and time_iters[ti_index] == step - warmup:
            times.append((len(xs), time.clock() - start_time))
            ti_index += 1

    accept_rate = accepts / (steps - warmup)
    if len(xs) > 1:
        if time_iters is not None:
            return xs, accept_rate, times
        else:
            return xs, accept_rate
    else:
        if time_iters is not None:
            return xs[0], accept_rate, times
        else:
            return xs[0], accept_rate
