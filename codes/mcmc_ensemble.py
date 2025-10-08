import numpy as np
from mcmc import metropolis_hastings, autocorrelation
from typing import Optional, Tuple, List

def generate_nearby_inits(theta_center, bounds, rng=None, n_chains=8, scale=0.1):
    """
    Generate n_chains initial thetas near theta_center, respecting bounds.
    scale: fraction of (bound range) used as std for perturbations.
    Returns array shape (n_chains, d).
    """
    if rng is None:
        rng = np.random.default_rng()
    theta_center = np.atleast_1d(theta_center).astype(float)
    d = theta_center.shape[0]
    inits = np.zeros((n_chains, d), dtype=float)

    # compute ranges
    ranges = np.array([b[1] - b[0] for b in bounds], dtype=float)

    for k in range(n_chains):
        th = theta_center.copy()

        # w: additive normal noise, then clip inside (0,1)
        # 0 and 1 are not included in the prior: Beta(2,2) -> inserts a small eps into the clip
        w_std = max(1e-3, scale * ranges[0])
        th[0] = rng.normal(th[0], w_std)
        th[0] = np.clip(th[0], bounds[0][0] + 1e-2, bounds[0][1] - 1e-2)

        for i in range(1, d):
            param_std = max(1e-3, scale * ranges[i])
            th[i] = rng.normal(th[i], param_std)
            th[i] = np.clip(th[i], bounds[i][0], bounds[i][1])

        inits[k, :] = th

    return inits

def run_ensemble_chains(theta_center, x, bounds, rng=None,
                        n_chains=8, n_steps=50000, cov_prop=None,
                        sigma_logT90=0.0, scale=0.1, verbose=True):
    """
    Generate n_chains initial points near theta_center and run metropolis_hastings
    for each chain. Returns dict with keys:
      'theta0s' : (n_chains, d) initial points
      'chains'  : list of arrays (n_steps, d)
      'acc_rates': list of empirical acceptance rates per chain
    cov_prop: initial proposal covariance passed to metropolis_hastings via init_cov.
              If None, metropolis_hastings will use its default small diag.
    """
    if rng is None:
        rng = np.random.default_rng()

    theta_center = np.atleast_1d(theta_center).astype(float)
    inits = generate_nearby_inits(theta_center, bounds, rng=rng, n_chains=n_chains, scale=scale)

    chains = []
    acc_rates = []

    for k in range(n_chains):
        theta0_k = inits[k].copy()
        if verbose:
            print(f"Starting chain {k+1}/{n_chains}, theta0 = {theta0_k}")

        # run the chain (metropolis_hastings prints progress and final acceptance)
        chain_samples = metropolis_hastings(theta0_k, x, bounds,
                                           init_cov=cov_prop, rng=rng,
                                           sigma_logT90=sigma_logT90, n=n_steps)
        # chain_samples shape (n_steps, d)
        chains.append(chain_samples)

        # compute empirical acceptance rate from samples: fraction of transitions where sample changed
        if chain_samples.shape[0] >= 2:
            diffs = np.any(chain_samples[1:, :] != chain_samples[:-1, :], axis=1)
            emp_acc = float(np.sum(diffs)) / float(diffs.size)
        else:
            emp_acc = 0.0
        acc_rates.append(emp_acc)
        if verbose:
            print(f"  empirical acceptance (from samples) = {emp_acc:.3f}")

    return {
        'theta0s': inits,
        'chains': chains,
        'acc_rates': acc_rates
    }

def thin_chains_by_acf(
    chains: np.ndarray,
    threshold: float = 0.05,
    max_lag_search: Optional[int] = None,
    autocorr_func = None,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Apply thinning to chains based on the first lag where the autocorrelation
    falls below `threshold`, evaluated separately for each parameter and then
    taking the maximum across parameters for each chain.

    Parameters
    ----------
    chains : np.ndarray
        Array of shape (n_chains, n_steps, d) or (n_steps, d) for a single chain.
    threshold : float
        Threshold on ACF (default 0.05).
    max_lag_search : int or None
        Maximum lag to check (if None => n_steps-1).
    autocorr_func : callable or None
        Function autocorrelation(x, norm=True) -> acf array. If None uses the
        `autocorrelation` function defined in mcmc.py file.

    Returns
    -------
    thinned_chains_list : list of np.ndarray
        List of length n_chains; each element is the thinned chain (shape (n_kept, d)).
    per_chain_param_lags : np.ndarray
        Array shape (n_chains, d) with the first lag >=1 where acf < threshold (if not found => max_lag_search).
    per_chain_thinning_lag : np.ndarray
        Array shape (n_chains,) with the effective step (max over params per chain), guaranteed >= 1.
    """

    # normalize input to (n_chains, n_steps, d)
    if chains.ndim == 2:
        chains = chains[np.newaxis, ...]  # (1, n_steps, d)
    elif chains.ndim != 3:
        raise ValueError("`chains` must be 2D (n_steps,d) or 3D (n_chains,n_steps,d)")

    n_chains, n_steps, d = chains.shape

    if max_lag_search is None:
        max_lag_search = max(1, n_steps - 1)  # do not consider lag 0 in the search

    # use your autocorrelation if none is passed
    if autocorr_func is None:
            autocorr_func = autocorrelation  # use the function defined in your module

    # output containers
    per_chain_param_lags   = np.zeros((n_chains, d), dtype=int)
    per_chain_thinning_lag = np.ones((n_chains,), dtype=int)
    thinned_indices        = []
    thinned_chains_list    = []

    # for each chain
    for k in range(n_chains):
        param_lags = np.zeros((d,), dtype=int)
        # for each parameter
        for j in range(d):
            x = chains[k, :, j]
            # compute full ACF with your function and truncate to max_lag_search
            acf = np.asarray(autocorr_func(x))
            # ensure we have at least lags 0..max_lag_search available
            max_available_lag = min(len(acf)-1, max_lag_search)
            # find first lag >= 1 with acf[lag] < threshold
            found = None
            for lag in range(1, max_available_lag + 1):
                if acf[lag] < threshold:
                    found = lag
                    break
            if found is None:
                # not found within max_lag_search -> use max_available_lag
                found = max_available_lag
            param_lags[j] = int(found)

        per_chain_param_lags[k, :] = param_lags
        # thinning step: maximum across param_lags (at least 1)
        step = int(max(1, param_lags.max()))
        per_chain_thinning_lag[k] = step
        thinned_chains_list.append(chains[k, ::step, :].copy())

    return thinned_chains_list, per_chain_param_lags, per_chain_thinning_lag
