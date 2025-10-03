import numpy as np
from mcmc import *

def estimate_covariance(theta0, x, bounds, rng=None,
                                  sigma_logT90=0.0,
                                  max_adapt=20000,
                                  adapt_interval=500,
                                  min_adapt=2000,
                                  tol_cov=1e-3,
                                  eps=1e-8,
                                  verbose=True):
    """
    Estimate empirical covariance through a short adaptive tuning phase.

    Parameters
    ----------
    theta0 : array_like
        Initial parameter vector (1D, length d).
    x : array_like
        Observed data passed to `log_posterior`.
    bounds : sequence
        Prior bounds passed to `log_posterior` (one (low, high) tuple per parameter).
    rng : RandomState or module-like, optional
        RNG object with methods `.multivariate_normal` and `.uniform`. If None, `np.random` is used.
    sigma_logT90 : float, optional
        Extra Gaussian noise parameter forwarded to `log_posterior`.
    max_adapt : int, optional
        Maximum number of adaptation iterations to run.
    adapt_interval : int, optional
        Every `adapt_interval` steps the function computes the classical sample covariance
        over all collected adaptive samples using `np.cov`.
    min_adapt : int, optional
        Minimum number of iterations before the stability test can stop adaptation early.
    tol_cov : float, optional
        Tolerance used to decide covariance stability. The relative Frobenius norm
        deltaC = ||C_new - C_prev||_F / ||C_prev||_F is compared to this threshold.
    eps : float, optional
        Small numerical jitter added to covariance estimates to ensure positive-definiteness.
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

    """
    if rng is None:
        rng = np.random

    theta = np.atleast_1d(theta0).astype(float)
    d = theta.shape[0]

    samples = []
    samples.append(theta.copy())
    logP = log_posterior(x, theta, bounds, sigma_logT90=sigma_logT90)
    accepted = 0

    cov_history = []
    cov_est = np.eye(d) * 1e-2

    for t in range(1, max_adapt+1):
        prop_cov = cov_est + eps * np.eye(d)
        theta_prop = rng.multivariate_normal(theta, prop_cov)
        logP_prop = log_posterior(x, theta_prop, bounds, sigma_logT90=sigma_logT90)

        if np.isfinite(logP_prop) and (logP_prop - logP > np.log(rng.uniform())):
            theta = theta_prop
            logP = logP_prop
            accepted += 1

        samples.append(theta.copy())

        # ogni adapt_interval calcola cov 
        if (t % adapt_interval) == 0:
            arr = np.asarray(samples)   # shape (n_samples, d)
            if arr.shape[0] > 1:
                cov_new = np.cov(arr, rowvar=False, bias=False) + eps * np.eye(d)
            else:
                cov_new = cov_est.copy()

            cov_history.append(cov_new.copy())

            # acceptance nella finestra (ultimi adapt_interval campioni)
            window = arr[-adapt_interval:,:] if arr.shape[0] >= adapt_interval else arr
            # per ottenere acceptance window dobbiamo contare cambi di stato; approssimiamo con
            # numero campioni unici: (semplice) -> meglio tenere buffer di accettazioni; per semplicità:
            # qui stimiamo acc_window come (accepted / t) su tutta la finestra: approssimazione
            acc_window = float(accepted) / float(t)  

            # check cov stability
            stable = False
            if len(cov_history) >= 2 and t >= min_adapt:
                C_now = cov_history[-1]
                C_prev = cov_history[-2]
                deltaC = np.linalg.norm(C_now - C_prev, ord='fro') / (np.linalg.norm(C_prev, ord='fro') + 1e-12)
                if verbose:
                    print(f"[adapt t={t}] overall_acc={accepted/float(t):.3f} acc_window≈{acc_window:.3f} deltaC={deltaC:.4e}")
                if deltaC < tol_cov:
                    stable = True

            cov_est = cov_new

            if stable:
                if verbose:
                    print(f"Adattamento stabile al passo {t}. Stop adattamento.")
                break

    samples = np.asarray(samples)
    overall_acc = accepted / float(max(1, len(samples)))
    mean_est = np.mean(samples, axis=0)
    
    return cov_est, mean_est, samples, overall_acc, cov_history