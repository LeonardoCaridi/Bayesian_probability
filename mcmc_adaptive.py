import numpy as np
from mcmc import *

def estimate_covariance(theta0, x, bounds, rng=None,
                        sigma_logT90=0.0,
                        max_adapt=100000,
                        adapt_interval=2000,
                        min_adapt=10000,
                        eps=1e-8,
                        tol_logdet=2e-2,
                        consecutive_stable=3,
                        verbose=True):
    """
    Estimate empirical covariance through a short adaptive tuning phase.

    Parameters
    ----------
    theta0 : array_like
        Initial parameter vector (1D, length d).
    x : array_like
        Observed data passed to `log_posterior`.
    bounds : list
        Prior bounds passed to `log_posterior` (one (low, high) tuple per parameter).
    rng : RandomState, optional
        RNG object with methods `.multivariate_normal` and `.uniform`. If None, `np.random` is used.
    sigma_logT90 : float, optional
        Extra Gaussian noise parameter forwarded to `log_posterior`.
    max_adapt : int, optional
        Maximum number of adaptation iterations to run.
    adapt_interval : int, optional
        Every `adapt_interval` steps the function computes the sample covariance
        over all collected adaptive samples using `np.cov`.
    min_adapt : int, optional
        Minimum number of iterations before the stability test can stop adaptation early.
    eps : float, optional
        Small numerical jitter added to covariance estimates to ensure positive-definiteness.
    tol_logdet : float, optional
        Threshold for |logdet(C_new) - logdet(C_prev)|. This is a log-volume difference;
        tol_logdet ~ 2e-2 means ~2% relative change in volume.
    consecutive_stable : int, optional
        Number of consecutive adapt-interval checks with |Δlogdet| < tol_logdet required to stop.
        Used to prevent false positive. The volume contraction is not monotonically decreasing and 
        may fluctuate around the tolerance. consecutive_stable allows you to verify that the variation is
        definitively below the tolerance.
    verbose : bool, optional
        If True, print progress messages during adaptation.

    Returns
    -------
    cov_est : ndarray, shape (d, d)
        Last estimated covariance matrix (includes `eps` jitter).
    mean_est : ndarray, shape (d,)
        Empirical mean of all samples collected during the adaptive phase.
    samples_adapt : ndarray, shape (n_samples, d)
        All samples collected during adaptation (first row is the initial `theta0`).
    overall_acc : float
        Overall acceptance rate during the adaptive phase, computed as accepted / n_samples.
    cov_history : list of ndarray
        List of covariance estimates computed at each `adapt_interval` step.
    vol_history: list
        List of volume contraction in %

    """

    if rng is None:
        rng = np.random

    theta = np.atleast_1d(theta0).astype(float)
    d = theta.shape[0]

    samples = []
    samples.append(theta.copy())
    logP = log_posterior(x, theta, bounds, sigma_logT90=sigma_logT90)

    accepted = 0
    accept_flags = [] # mask of accepted samples

    cov_history = []
    cov_est = np.eye(d) * 1e-2  # initial small scale

    vol_history = []

    stable_count = 0

    for t in range(1, max_adapt+1):
        # proposal covariance 
        #prop_cov = cov_est + eps * np.eye(d)   # cov_est already includes eps
        theta_prop = rng.multivariate_normal(theta, cov_est)
        logP_prop = log_posterior(x, theta_prop, bounds, sigma_logT90=sigma_logT90)

        accepted_flag = 0
        if (logP_prop - logP > np.log(rng.uniform())):
            theta = theta_prop
            logP = logP_prop
            accepted += 1
            accepted_flag = 1

        accept_flags.append(accepted_flag)
        samples.append(theta.copy())

        # every adapt_interval iterations compute the covariance on all collected samples
        if (t % adapt_interval) == 0:
            arr = np.asarray(samples)   # shape (n_samples, d)

            if arr.shape[0] > 1:
                # compute sample covariance once and add eps (jitter) exactly here
                cov_new = np.cov(arr, rowvar=False, bias=False)
                cov_new = cov_new + eps * np.eye(d)   # add jitter 
            else:
                cov_new = cov_est.copy()

            # store the regularized covariance
            cov_history.append(cov_new.copy())

            # exact acceptance in the last window of length adapt_interval (or shorter if not enough samples)
            if len(accept_flags) >= adapt_interval:
                window_flags = accept_flags[-adapt_interval:]
            else:
                window_flags = accept_flags[:]
            acc_window = float(sum(window_flags)) / float(len(window_flags)) if len(window_flags) > 0 else 0.0
            overall_acc = float(sum(accept_flags)) / float(len(accept_flags))

            # compute log-determinant difference using covariances already regularized in cov_history
            stable = False
            if len(cov_history) >= 2 and t >= min_adapt:
                C_now = cov_history[-1]   # already includes eps
                C_prev = cov_history[-2]  # already includes eps

                # compute slogdet directly on stored matrices
                sign_now, logdet_now = np.linalg.slogdet(C_now)
                sign_prev, logdet_prev = np.linalg.slogdet(C_prev)
                
                delta_logdet = float(logdet_now - logdet_prev)  # log difference (volume ratio)

                vol_ratio = np.exp(delta_logdet)
                vol_pct = (vol_ratio - 1.0) * 100.0
                vol_history.append(vol_pct)
                
                if verbose:
                    print(f"[adapt t={t}] overall_acc={overall_acc:.3f} acc_window={acc_window:.3f} "
                          f"delta_logdet={delta_logdet:.4e} (~{vol_pct:.3f}%) stable_count={stable_count}")

                # STOP criterion: based on log-determinant change
                if abs(delta_logdet) < tol_logdet:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= consecutive_stable:
                    stable = True

            # update cov_est for next proposals
            cov_est = cov_new

            if stable:
                if verbose:
                    print(f"Covariance stabilized by logdet at step {t} (stable_count={stable_count}). Stopping adaptation.")
                break

    samples = np.asarray(samples)
    overall_acc = float(sum(accept_flags)) / float(max(1, len(accept_flags)))
    mean_est = np.mean(samples, axis=0)

    return cov_est, mean_est, samples, overall_acc, cov_history, vol_history