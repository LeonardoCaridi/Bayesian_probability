from typing import Tuple, Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

def classification_probability(
    x: float,
    samples: np.ndarray,
    name: str = None,
    sigma_logT90: float = 0.0,
    model: str = 'short',
    save: bool = True,
    show: bool = True
) -> np.ndarray:
    """
    Plot the posterior distribution (from MCMC samples) of the probability that
    a GRB with observed logT90 == x belongs to the 'short' (or 'long') class.
    
    Parameters
    ----------
    x: float
        The ln(T90) value to classify (log(s))
    samples : np.ndarray
        2D array of MCMC samples after burn-in and thinning. Each row is one
        sample and the expected column ordering is:
            samples[:,0] = w                (mixing weight for the short population)
            samples[:,1] = mu_s             (mean of short population)
            samples[:,2] = mu_l - mu_s      (difference so that mu_l = mu_s + delta)
            samples[:,3] = log(sigma_s)     (log sigma short)
            samples[:,4] = log(sigma_l)     (log sigma long)
    name: str
        Name used in the filenames and title. If None, a default name based on x 
        is created: f'logT90_{x:.3f}'.
    sigma_logT90: float
        Gaussian observational uncertainty on x (in the same units: log(s)).
        Default 0.0 (no measurement error).
    model : {'short','long','both'}, optional
        Which histogram(s) to plot:
            - 'short': only P(short | x, θ_i) histogram
            - 'long' : only P(long  | x, θ_i) histogram
            - 'both' : overlay both histograms (P(short) and P(long))
    save: bool
        Save the plot at filepath
    show: bool
        Show the plot

    Return
    ------
    h or g or [h,g]: np.ndarray
        probability distribution
    """
    if model not in ['short', 'long', 'both']:
        raise ValueError("'model' must be 'short', 'long' or 'both'")
    if name is None:
        name = f'ln({int(np.exp(x))})'
    
    if sigma_logT90 == 0.0:
        filepath = f'../figure/classification/{name}_{model}_probability'
        value    = 'x'
    else:
        filepath = f'../figure/classification/{name}_{model}_probability_with_sigma'
        value    = '$x_{obs}$'
        
    # unpack parameters from samples
    w       = samples[:,0]
    mu_s    = samples[:,1]
    mu_l    = samples[:,1] + samples[:,2]
    sigma_s = np.exp(samples[:,3])
    sigma_l = np.exp(samples[:,4])
    
    # include measurement error in variance
    var_s = sigma_s**2 + sigma_logT90**2 
    var_l = sigma_l**2 + sigma_logT90**2 
        
     # log-likelihoods for the two Gaussians (vectorized across samples)
    logN1 = -0.5*(np.log(2*np.pi*var_s) + ((x-mu_s)**2)/var_s)
    logN2 = -0.5*(np.log(2*np.pi*var_l) + ((x-mu_l)**2)/var_l)
    
    # numerically stable normalization: log posterior marginal p(x | θ)
    stacked  = np.vstack([np.log(w) + logN1, np.log(1-w) + logN2]) 
    log_post = logsumexp(stacked, axis=0) 
    
    # posterior probability of "short" for each sample θ_i
    h = np.exp(np.log(w)+logN1-log_post)

     # bins: if both, force range [0,1] so histograms line up
    if model == 'both': bins = np.linspace(0,1,51)
    else: bins = 50 # integer -> matplotlib decides bin edges automatically
    
    if model in ('short', 'both'):
        mean_h     = np.mean(h)
        median_h   = np.median(h)
        p5_h       = np.percentile(h,5)
        p95_h      = np.percentile(h,95)

        if show:
            plt.hist(h, bins=bins, density=True, alpha=0.55, edgecolor='k')
            plt.axvline(mean_h, color='red', linestyle='-', label=f'mean={mean_h:.3f}', linewidth = 3)
            plt.axvline(median_h, color='orange', linestyle='--', label=f'median={median_h:.3f}', linewidth = 3)
            plt.axvline(p5_h, color='grey', linestyle=':', label=f'5%={p5_h:.3f}', linewidth = 2.5)
            plt.axvline(p95_h, color='grey', linestyle=':', label=f'95%={p95_h:.3f}', linewidth = 2.5)
        
    if model in ('long', 'both'):
        g        = 1 - h
        mean_g   = np.mean(g)
        median_g = np.median(g)
        p5_g     = np.percentile(g,5)
        p95_g    = np.percentile(g,95)

        if show:
            plt.hist(g, bins=bins, density=True, alpha=0.55, edgecolor='k')
            plt.axvline(mean_g, color='blue', linestyle='-', label=f'mean={mean_g:.3f}', linewidth = 3)
            plt.axvline(median_g, color='green', linestyle='--', label=f'median={median_g:.3f}', linewidth = 3)
            plt.axvline(p5_g, color='darkgrey', linestyle=':', label=f'5%={p5_g:.3f}', linewidth = 2.5)
            plt.axvline(p95_g, color='darkgrey', linestyle=':', label=f'95%={p95_g:.3f}', linewidth = 2.5)    
    if show:
        if model == 'both':
            xlabel = 'Probability'
            title  = f'Histogram of P(short| {value}, $θ$) and P(long| {value}, $θ$) for x={name} (x={x:.3f})'
        else:
            xlabel = f'P({model} | {value}, $\\theta_i$)'
            title  = f'Histogram of P({model}| {value}, $θ$) for x={name} (x={x:.3f})'
            
        plt.xlabel(xlabel)
        plt.ylabel('Density')
        plt.title(title)
        
        plt.legend()
    
        if save:
            plt.savefig(filepath + '.png', dpi=300, bbox_inches='tight')
            plt.savefig(filepath + '.pdf', dpi=300, bbox_inches='tight')
        plt.show()
    if model == 'short': return h
    elif model == 'long': return g
    else: return np.array([h,g])

def crossing(
    x: np.array,
    y: np.array,
    level: float = 0.5
) -> float:
    """
    Find the first crossing point x where y(x) == level using linear interpolation.

    Method
    ------
    - Compute s = y - level and look for the first index k where s changes sign
      (i.e. np.diff(np.sign(s)) != 0). If no sign change is found, returns np.nan.
    - Performs a linear interpolation between (x[k], y[k]) and (x[k+1], y[k+1])
      to estimate the x value where y == level:
          x_cross = x0 + (level - y0) * (x1 - x0) / (y1 - y0)

    Parameters
    ----------
    x : 1D array_like
        Monotonic grid of abscissa values where y was evaluated (must be same length as y).
    y : 1D array_like
        Function values evaluated on x.
    level : float, optional (default=0.5)
        The target horizontal level for which to find the crossing.

    Returns
    -------
    float
        Interpolated x coordinate of the first crossing. Returns np.nan if no crossing is found.
    """
    s = y - level
    sign = np.sign(s)
    # np.diff(sign) computes sign[i+1] - sign[i].
    # (np.diff(sign) != 0) detects changes in the sign array (including transitions to/from 0).
    idxs = np.where(np.diff(sign) != 0)[0]
    if idxs.size == 0:
        return np.nan
    k = idxs[0]
    x0, x1 = x[k], x[k+1]
    y0, y1 = y[k], y[k+1]
    # linear interpolation
    return x0 + (level - y0)*(x1-x0)/(y1-y0)

def estimate_threshold(
    samples: np.ndarray,
    x_grid: np.array = None
) -> float:
    """
    Estimate a decision threshold (in ln(T90)) by solving mean_i P(short | x, theta_i) == 0.5.

    Description
    -----------
    For each x in a grid, compute the posterior-average probability
        h(x) = E_theta[ P(short | x, theta) ] ≈ (1/N) sum_i P(short | x, theta_i),
    then find the x where h(x) crosses 0.5 (using `crossing` for linear interpolation).
    The returned threshold is the solution of mean_i P(short | x, theta_i) = 0.5.

    Parameters
    ----------
    samples : ndarray, shape (N, 5)
        MCMC samples after burn-in/thinning. (see `classification_probability` for column order).
    x_grid : 1D array_like, optional
        Grid of ln(T90) values where h(x) will be evaluated. If None, defaults to
        np.linspace(-4, 7, 10001).

    Returns
    -------
    float
        Estimated threshold (ln(T90)) where the posterior-average probability of "short"
        equals 0.5 (interpolated).
    """
    if x_grid is None:
        x_grid = np.linspace(-4,7,10001)
    x = x_grid.astype(np.float64)
    h_means = []

    for i in range(x.shape[0]):
        h = classification_probability(x[i], samples, model='short', save=False, show=False)
        h_means.append(np.mean(h))

    h_means = np.asarray(h_means)

    thr_mean_h = crossing(x, h_means, 0.5)
    return thr_mean_h

def estimate_threshold_distribution(
    samples: np.ndarray, 
    x_grid: np.array = None,
    sigma_logT90 = 0.0,
    show: bool = True,
    save: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    Compute the posterior distribution of thresholds by solving P(short | x, theta_i) == 0.5
    for each MCMC sample theta_i.

    Description
    -----------
    For each sample theta_i:
      - evaluate h_i(x) = P(short | x, theta_i) on x_grid (optionally adding observational
        variance sigma_logT90**2 to each component variance),
      - find the first x where h_i(x) == 0.5 (using `crossing`),
      - collect thr_i into an array thr_arr.
    The function returns thr_arr and summary statistics (median, mean, std, 5-95% interval).

    Parameters
    ----------
    samples : ndarray, shape (N,5)
        MCMC samples (see `classification_probability` for column order).
    x_grid : 1D array_like, optional
        Grid of ln(T90) values for evaluating h_i(x). Defaults to np.linspace(-4,7,10001).
    sigma_logT90 : float, optional (default=0.0)
        Additive measurement uncertainty in ln(T90). For each component use
        var = sigma_component**2 + sigma_logT90**2 when computing h_i(x).
        (If you want to reflect true observational noise, pass the real measurement sigma here.)
    show : bool, default True
        If True, plot a histogram of thr_arr with summary lines.
    save : bool, default True
        If True, save the histogram

    Returns
    -------
    thr_arr : ndarray, shape (N,)
        Array of per-sample thresholds (NaN for samples that have no crossing on x_grid).
    info : dict
        Summary statistics:
            {'median': ..., 'mean': ..., 'std': ..., '[5,95]%': [p5, p95]}
    """
    if x_grid is None:
        x_grid = np.linspace(-4,7,10001)
    x = x_grid.astype(np.float64)
    thr_list = []
    
    # unpack samples 
    w = samples[:,0]
    mu_s = samples[:,1]
    mu_l = samples[:,1] + samples[:,2]
    sigma_s = np.exp(samples[:,3])
    sigma_l = np.exp(samples[:,4])

    for i in range(samples.shape[0]):
        # vectorized across x (not samples)
        var_s = sigma_s[i]**2 + sigma_logT90**2
        var_l = sigma_l[i]**2 + sigma_logT90**2
        logN1 = -0.5*(np.log(2*np.pi*var_s) + ((x - mu_s[i])**2)/var_s)
        logN2 = -0.5*(np.log(2*np.pi*var_l) + ((x - mu_l[i])**2)/var_l)
        logpost = logsumexp([np.log(w[i]) + logN1, np.log(1-w[i]) + logN2], axis=0)
        h = np.exp(np.log(w[i]) + logN1 - logpost)
        
        # crossing interpolation
        thr_i = crossing(x, h, 0.5)
        thr_list.append(thr_i)
        
    thr_arr = np.array(thr_list)
    thr_median = np.nanmedian(thr_arr)
    thr_mean = np.nanmean(thr_arr)
    thr_std = np.nanstd(thr_arr, ddof=1)
    p5_thr, p95_thr = np.nanpercentile(thr_arr, [5,95])
    info = {'median': thr_median, 'mean': thr_mean, 'std': thr_std, '[5,95]%': [p5_thr, p95_thr]}

    if show:
        plt.hist(thr_arr, bins=50, density=True, alpha=0.55, edgecolor='k')
        plt.axvline(thr_mean, linestyle='-', color='r', linewidth=3, label=f'mean={thr_mean:.3f}')
        plt.axvline(thr_median, linestyle='--', color='orange', linewidth=3, label=f'median={thr_median:.3f}')
        plt.axvline(p5_thr, linestyle=':', color='gray', linewidth=2.5, label=f'5%={p5_thr:.3f}')
        plt.axvline(p95_thr, linestyle=':', color='gray', linewidth=2.5, label=f'95%={p95_thr:.3f}')
        plt.xlabel('ln(T90)')
        plt.ylabel('Density')
        plt.title('Threshold distribution')
        plt.legend()
        if save:
            filepath = f'../figure/classification/thr_distribution'
            plt.savefig(filepath + '.png', dpi=300, bbox_inches='tight')
            plt.savefig(filepath + '.pdf', dpi=300, bbox_inches='tight')
        plt.show()
        
    return thr_arr, info

def distribution_probability_threshold(
    samples: np.ndarray,
    thr_arr: np.array,
    save: bool = True
):
    """
    Evaluate and plot the distribution of probabilities when the threshold
    is varied across thr_arr.

    Description
    -----------
    Given a set of candidate thresholds thr_arr, for each threshold x compute:
        h_mean(x) = (1/N) * sum_i P(short | x, theta_i),
        g_mean(x) = (1/N) * sum_i P(long  | x, theta_i).
    The function builds histograms of h_mean and g_mean over the values in thr_arr,
    and plots summary lines (mean, median, 5% and 95%).

    Parameters
    ----------
    samples : ndarray, shape (N,5)
        MCMC samples (see `classification_probability` for column order).
    thr_arr : 1D array_like
        Candidate threshold values (ln(T90)). NaNs will be removed.
    save : bool, default True
        If True, save the produced figure to disk (path is hard-coded in the implementation).
    """
    # clean thr_arr
    thr_arr = np.asarray(thr_arr)
    thr_arr = thr_arr[~np.isnan(thr_arr)]
    if thr_arr.size == 0:
        raise ValueError("thr_arr is empty after dropping NaNs")

    # Is possibile to vectorize for faster computation (not implemented)
    h_means = []
    g_means = []
    for x in thr_arr:
        h = classification_probability(x, samples, model='short', save=False, show=False)
        g = 1-h
        h_means.append(np.mean(h))
        g_means.append(np.mean(g))
    h_means = np.asarray(h_means)
    g_means = np.asarray(g_means)    

    bins = np.linspace(0,1,51)
    plt.hist(h_means, bins=bins, density=True, alpha=0.55, edgecolor='k')
    plt.hist(g_means, bins=bins, density=True, alpha=0.55, edgecolor='k')
    
    mean_h     = np.mean(h_means)
    median_h   = np.median(h_means)
    p5_h       = np.percentile(h_means,5)
    p95_h      = np.percentile(h_means,95)
    
    plt.axvline(mean_h, color='red', linestyle='-', label=f'mean={mean_h:.3f}', linewidth = 3)
    plt.axvline(median_h, color='orange', linestyle='--', label=f'median={median_h:.3f}', linewidth = 3)
    plt.axvline(p5_h, color='grey', linestyle=':', label=f'5%={p5_h:.3f}', linewidth = 2.5)
    plt.axvline(p95_h, color='grey', linestyle=':', label=f'95%={p95_h:.3f}', linewidth = 2.5)
    
    mean_g   = np.mean(g_means)
    median_g = np.median(g_means)
    p5_g     = np.percentile(g_means,5)
    p95_g    = np.percentile(g_means,95)
    
    plt.axvline(mean_g, color='blue', linestyle='-', label=f'mean={mean_g:.3f}', linewidth = 3)
    plt.axvline(median_g, color='green', linestyle='--', label=f'median={median_g:.3f}', linewidth = 3)
    plt.axvline(p5_g, color='darkgrey', linestyle=':', label=f'5%={p5_g:.3f}', linewidth = 2.5)
    plt.axvline(p95_g, color='darkgrey', linestyle=':', label=f'95%={p95_g:.3f}', linewidth = 2.5)   

    plt.xlabel('Probability at varying threshold')
    plt.ylabel('Density')
    plt.title(r'Distribution of $\frac{1}{N} \sum_i$P(short| $thr_j$, $θ_i$) and $\frac{1}{N} \sum_i$P(long| $thr_j$, $θ_i$)')
    
    plt.legend()
    if save:
        filepath = f'../figure/classification/thr_distribution_probability'
        plt.savefig(filepath + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(filepath + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()





