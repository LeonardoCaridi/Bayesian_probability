from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt


def trace_plot(samples: np.ndarray, param_names: Optional[Sequence[str]] = None, figsize: Optional[tuple] = None, true_values: Optional[Sequence[float]] = None):
    """Plot trace (time series) for each parameter.

    Parameters
    ----------
    samples : ndarray, shape (d, N)
        Samples where each row is a parameter and columns are iterations.
    param_names : sequence of str, optional
        Names for parameters (length d). If None generic names param_0,... will be used.
    figsize : tuple, optional
        Figure size. If None a sensible default is used depending on d.
    true_values : sequence of float or None
        If provided, should have length d. For each parameter a red dashed
        horizontal line will be drawn at the true parameter value.
    """
    d, N = samples.shape
    if param_names is None:
        param_names = [f"param_{i}" for i in range(d)]

    if true_values is not None and len(true_values) != d:
        raise ValueError("true_values must have length equal to number of parameters (d)")

    if figsize is None:
        figsize = (10, max(2.0 * d, 4))

    fig, axes = plt.subplots(d, 1, figsize=figsize, sharex=True)
    if d == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(np.arange(N), samples[i, :], linewidth=1.5)
        ax.set_ylabel(param_names[i])
        ax.grid(True, linestyle=':', alpha=0.6)
        if true_values is not None:
            ax.axhline(true_values[i], color='red', linestyle='--', linewidth=2)
    axes[-1].set_xlabel('Iteration')
    fig.tight_layout()
    return fig, axes



def marginal_plots(samples: np.ndarray, param_names: Optional[Sequence[str]] = None, bins: int = 30, figsize: Optional[tuple] = None, true_values: Optional[Sequence[float]] = None):
    """Plot marginal histograms for each parameter.

    Also draws the 5th, 50th (median) and 90th percentiles on each histogram and
    displays a legend identifying them.

    Parameters
    ----------
    samples : ndarray, shape (d, N)
    param_names : sequence of str, optional
    bins : int
    figsize : tuple, optional
    true_values : sequence of float or None
        If provided, should have length d. A red dashed vertical line will be drawn
        on each marginal histogram at the corresponding true value.
    """
    d, N = samples.shape
    if param_names is None:
        param_names = [f"param_{i}" for i in range(d)]

    if true_values is not None and len(true_values) != d:
        raise ValueError("true_values must have length equal to number of parameters (d)")

    ncols = min(3, d)
    nrows = int(np.ceil(d / ncols))
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    for i in range(d):
        ax = axes[i]
        # histogram
        counts, bins_edges, patches = ax.hist(samples[i, :], bins=bins, density=True, edgecolor='k', alpha=0.7)
        ax.set_title(param_names[i])
        ax.grid(True, linestyle=':', alpha=0.5)

        # compute percentiles
        q5, q50, q95 = np.percentile(samples[i, :], [5, 50, 95])

        # draw percentile lines with labels for legend
        line_p5 = ax.axvline(q5, color='gray', linestyle=':', linewidth=2, label='5th pctile')
        line_p50 = ax.axvline(q50, color='gray', linestyle='--', linewidth=2.5, label='median')
        line_p95 = ax.axvline(q95, color='gray', linestyle=':', linewidth=2, label='95th pctile')

        # if true value provided, draw it too
        if true_values is not None:
            line_true = ax.axvline(true_values[i], color='red', linestyle='-', linewidth=3, label='true value')
            txt = f"med={q50:.3g}\n5%={q5:.3g}\n95%={q95:.3g}\nmcmc={true_values[i]:.3g}"
            # ensure legend shows true value first (optional)
            #handles = [line_true, line_p5, line_p50, line_p90]
            #labels = [h.get_label() for h in handles]
            #ax.legend(handles, labels, fontsize='small')
        else:
            txt = f"med={q50:.3g}\n5%={q5:.3g}\n95%={q95:.3g}"
            #txt = f"med={q50:.3g}\n5–95%=[{q5:.3g}, {q95:.3g}]"
            #ax.legend(fontsize='small')
        ax.text(0.98, 0.95, txt, transform=ax.transAxes,
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white', alpha=0.95, edgecolor='none'),
                zorder=20)

    # hide unused axes
    for j in range(d, axes.size):
        axes[j].axis('off')

    fig.tight_layout()
    plt.suptitle("Gibbs sampling", y=1.03, fontsize=20)
    return fig, axes[:d]


def _autocorr_series(x: np.ndarray, nlags: int):
    """Compute autocorrelation function up to nlags using unbiased estimator.

    Returns array of length nlags+1 where index 0 is lag 0 (autocorr=1).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return np.zeros(nlags + 1)
    x = x - x.mean()
    # full autocov via correlate
    acov = np.correlate(x, x, mode='full')[n - 1: n - 1 + nlags + 1]
    # unbiased normalization: divide by (n - lag)
    denom = np.arange(n, n - nlags - 1, -1)
    acov = acov / denom
    # normalize by variance (lag 0)
    acov0 = acov[0]
    if acov0 == 0:
        return np.zeros(nlags + 1)
    return acov / acov0


def autocorr_plot(samples: np.ndarray, param_names: Optional[Sequence[str]] = None, nlags: int = 40, figsize: Optional[tuple] = None):
    """Plot autocorrelation up to nlags for each parameter.

    Parameters
    ----------
    samples : ndarray, shape (d, N)
    param_names : sequence of str, optional
    nlags : int
    figsize : tuple, optional
    """
    d, N = samples.shape
    if param_names is None:
        param_names = [f"param_{i}" for i in range(d)]

    if figsize is None:
        figsize = (10, max(2.0 * d, 4))

    fig, axes = plt.subplots(d, 1, figsize=figsize, sharex=True)
    if d == 1:
        axes = [axes]

    lags = np.arange(0, nlags + 1)
    for i, ax in enumerate(axes):
        acf = _autocorr_series(samples[i, :], nlags)
        ax.stem(lags, acf)
        ax.set_ylim(-1.0, 1.0)
        ax.set_ylabel(param_names[i])
        ax.grid(True, linestyle=':', alpha=0.6)
    axes[-1].set_xlabel('Lag')
    fig.tight_layout()
    return fig, axes

