import os
import numpy as np
import matplotlib.pyplot as plt
from dynesty import plotting as dyplot
from typing import Optional, Sequence, Tuple, Union, Dict

def cornerplot_ns(res, 
                  filename,
                  truths=None, 
                  model='H0',  
                  outdir='../figure/nested_sampling/', 
                  save=True):

    if model=='H0':
        ndim = 5

        # crea figura 5x5
        fig, axes = plt.subplots(ndim, ndim, figsize=(14, 14))
        axes = axes.reshape((ndim, ndim))
        
        labels = ['w', '$\mu_1$', '$\delta = \mu_2 - \mu_1$', '$ln\sigma_1$', '$ln\sigma_2$']
    elif model=='H1':
        ndim = 8

        # crea figura 5x5
        fig, axes = plt.subplots(ndim, ndim, figsize=(14, 14))
        axes = axes.reshape((ndim, ndim))
        
        labels = ['w1', 'w2', '$\mu_1$', '$\delta_{12} = \mu_2 - \mu_1$', '$\delta_{23} = \mu_3 - \mu_2$', 
                  '$ln\sigma_1$', '$ln\sigma_2$', '$ln\sigma_3$']
    else:
        raise ValueError("model must be 'H0' or 'H1'")
        
    fg, ax = dyplot.cornerplot(
        res,        
        color='blue',
        truths=truths,
        truth_color='red',
        show_titles=True,
        max_n_ticks=3,
        quantiles=[0.05,0.5,0.95],
        quantiles_2d=[0.05,0.5,0.95],
        labels = labels,
        fig=(fig, axes)         
    )
    
    # aggiusta spaziatura e mostra
    plt.subplots_adjust(wspace=0.03, hspace=0.03)
    if save:
        plt.savefig(outdir+filename+'.png', bbox_inches='tight', dpi=300)
        plt.savefig(outdir+filename+'.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    return fig, axes


def _theta_to_params(theta: np.ndarray):
    """
    Convert a theta vector into (weights, mus, sigs).
    Supports H0 (len=5) and H1 (len=8).
    Returns (weights, mus, sigs).
    """
    theta = np.asarray(theta)
    if theta.size == 5:
        # H0: (v, mu1, delta, ln_s1, ln_s2)
        v, mu1, delta, ln_s1, ln_s2 = theta
        weights = np.array([v, 1.0 - v])
        mus = np.array([mu1, mu1 + delta])
        sigs = np.exp(np.array([ln_s1, ln_s2]))
    elif theta.size == 8:
        # H1: (v1, v2, mu1, delta12, delta23, ln_s1, ln_s2, ln_s3)
        v1, v2, mu1, delta12, delta23, ln_s1, ln_s2, ln_s3 = theta
        w1 = v1
        w2 = (1.0 - v1) * v2
        w3 = (1.0 - v1) * (1.0 - v2)
        weights = np.array([w1, w2, w3])
        mu2 = mu1 + delta12
        mu3 = mu2 + delta23
        mus = np.array([mu1, mu2, mu3])
        sigs = np.exp(np.array([ln_s1, ln_s2, ln_s3]))
    else:
        raise ValueError("theta length must be 5 (H0) or 8 (H1)")
    return weights, mus, sigs


def plot_distribution_ns(
    sampler,
    logT90: np.ndarray,
    filename: str,
    samples: np.ndarray,
    samples2: Optional[np.ndarray] = None,
    theta_true: Dict = None,
    percentiles: Sequence[int] = (5, 50, 95),
    outdir: str = "../figure/nested_sampling/",
    save: bool = True,
    figsize: Tuple[int, int] = (8, 5),
    x_grid: Optional[np.ndarray] = None,
    bins: int = 30,
    label1: str = "H0",
    label2: str = "H1",
):
    """
    Function to plot posterior predictive distributions for logT90
    comparing up to two models built on the same data.

    - sampler: an instance exposing logT90_distribution(weights, mus, sigs, x=..., sigma_logT90=...)
    - samples: array shape (n_samples, n_params) for model 1 (n_params = 5 or 8)
    - samples2: optional array for model 2 (same shape rule)
    - theta_true: optional true params vector (applied to whichever model's dimension matches)
    """

    logT90 = np.asarray(logT90)
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("`samples` must be a 2D array (n_samples, n_params)")

    n_samps1, D1 = samples.shape
    if D1 not in (5, 8):
        raise ValueError("samples must have 5 (H0) or 8 (H1) columns")

    if samples2 is not None:
        samples2 = np.asarray(samples2)
        if samples2.ndim != 2:
            raise ValueError("`samples2` must be a 2D array (n_samples, n_params)")
        n_samps2, D2 = samples2.shape
        if D2 not in (5, 8):
            raise ValueError("samples2 must have 5 (H0) or 8 (H1) columns")

    # build x-grid
    if x_grid is None:
        x_min, x_max = np.min(logT90), np.max(logT90)
        x = np.linspace(x_min, x_max, 1000)
    else:
        x = np.asarray(x_grid)

    # evaluate model 1
    dens1 = np.empty((n_samps1, x.size))
    for i in range(n_samps1):
        theta = samples[i]
        weights, mus, sigs = _theta_to_params(theta)
        # force sigma_logT90 = 0.0 
        logd = sampler.logT90_distribution(weights, mus, sigs, x=x, sigma_logT90=0.0)
        dens1[i, :] = np.exp(logd)

    low1, med1, high1 = np.percentile(dens1, percentiles, axis=0)  # shape (len(percentiles), len(x))

    # evaluate model 2 if provided
    if samples2 is not None:
        dens2 = np.empty((n_samps2, x.size))
        for i in range(n_samps2):
            theta = samples2[i]
            weights, mus, sigs = _theta_to_params(theta)
            logd = sampler.logT90_distribution(weights, mus, sigs, x=x, sigma_logT90=0.0)
            dens2[i, :] = np.exp(logd)
        low2, med2, high2 = np.percentile(dens2, percentiles, axis=0)

    # prepare real curve if theta_true is provided
    real = None
    if theta_true is not None:
        w, m, s = theta_true['w'], theta_true['mus'], theta_true['sigmas']
        real = np.exp(sampler.logT90_distribution(w, m, s, x=x, sigma_logT90=0.0))

    # sorting for plotting (if xgrid is provided unordered, i.e. xgrid=logT90)
    order = np.argsort(x)
    xs = x[order]
    low1_s, med1_s, high1_s = low1[order], med1[order], high1[order]
    if samples2 is not None:
        low2_s, med2_s, high2_s = low2[order], med2[order], high2[order]

    # plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(logT90, bins=bins, density=True, alpha=0.35,
            facecolor='lightgrey', edgecolor='none', label='logT90 histogram', zorder=0)

    # model1 (blue-ish band + black median)
    ax.fill_between(xs, low1_s, high1_s, alpha=0.3, facecolor='C0',
                    label=f'{label1} {percentiles[0]}-{percentiles[-1]} pct', zorder=1)
    ax.plot(xs, med1_s, color='C0', linestyle='-', linewidth=1.6, label=f'{label1} median', zorder=3)

    # model2 (if present) (different color)
    if samples2 is not None:
        ax.fill_between(xs, low2_s, high2_s, alpha=0.25, facecolor='C1',
                        label=f'{label2} {percentiles[0]}-{percentiles[-1]} pct', zorder=2)
        ax.plot(xs, med2_s, color='C1', linestyle='-', linewidth=1.6, label=f'{label2} median', zorder=4)

    # real curves (single theta_true may have produced real1 and/or real2)
    if real is not None:
        ax.plot(xs, real[order], 'r--', label='Real', zorder=5)

    ax.set_xlabel('logT90')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()

    if save:
        png = os.path.join(outdir, filename + ".png")
        pdf = os.path.join(outdir, filename + ".pdf")
        fig.savefig(png, bbox_inches='tight', dpi=300)
        fig.savefig(pdf, bbox_inches='tight', dpi=300)

    plt.show()
    return fig, ax
