from typing import Optional, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import cm, colors


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

def plot_clustered_1d(data, assignment, figsize=(7,5), bins=30, alpha=0.6, title=None):
    """
    Plot histograms for 1D clustered data.

    Assumptions:
      - data is a numpy array with shape (n,)
      - assignment is a numpy array with shape (n,) containing cluster labels

    Parameters:
      data       : 1D numpy array of data points
      assignment : 1D numpy array of cluster labels (numeric or convertible to numeric)
      figsize    : figure size passed to plt.subplots
      bins       : number of histogram bins (passed to ax.hist)
      alpha      : transparency for histogram bars
      title      : optional title string for the axes

    Returns:
      matplotlib.figure.Figure
    """
    # use the provided arrays directly (assumed correct shape)
    X = data  # shape (n,)
    assignment = np.asarray(assignment)
    if X.shape[0] != assignment.shape[0]:
        raise ValueError("data and assignment must have same number of points.")
    
    # get unique cluster labels and count them
    unique_labels = np.unique(assignment)
    K = unique_labels.size

    # create figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # plot one histogram per cluster
    for lab in unique_labels:
        vals = X[assignment == lab]
        if vals.size == 0:
            continue
        # density=False keeps absolute counts; label shows cluster id
        ax.hist(vals, bins=bins, alpha=alpha, label=f"cluster {int(lab)}", density=False)

    # labels, title and legend
    ax.set_xlabel("logT90")
    ax.set_ylabel("count")
    ax.set_title(title if title is not None else f"1D clustered histogram — {K} clusters")
    ax.legend(title="Clusters", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    return fig


def plot_clustered_2d(data, assignment, figsize=(7,5), s=30, alpha=0.6, cmap_name='tab10', title=None):
    """
    Plot scatter for 2D clustered data.

    Assumptions:
      - data is a numpy array with shape (n, 2)
      - assignment is a numpy array with shape (n,) containing cluster labels

    Parameters:
      data       : 2D numpy array of shape (n,2) with columns [x, y]
      assignment : 1D numpy array of cluster labels (numeric or convertible to numeric)
      figsize    : figure size passed to plt.subplots
      s          : marker size for scatter
      alpha      : transparency for scatter points
      cmap_name  : name of matplotlib colormap to use for discrete clusters
      title      : optional title string for the axes

    Returns:
      matplotlib.figure.Figure
    """
    # use the provided arrays directly (assumed correct shape)
    X = data  # shape (n,2)
    assignment = np.asarray(assignment)
    if X.shape[0] != assignment.shape[0]:
        raise ValueError("data and assignment must have same number of points.")
    
    # unique labels and count
    unique_labels = np.unique(assignment)
    K = unique_labels.size

    # prepare figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # discrete colormap: normalize labels to colormap range
    cmap = cm.get_cmap(cmap_name)
    norm = colors.Normalize(vmin=unique_labels.min(), vmax=unique_labels.max())

    # scatter plot: color is taken from assignment via cmap+norm
    scatter = ax.scatter(X[:,0], X[:,1], c=assignment, cmap=cmap, norm=norm, s=s, alpha=alpha)

    # axis labels and title
    ax.set_xlabel("logT90")
    ax.set_ylabel("HR")
    ax.set_title(title if title is not None else f"2D clustered scatter — {K} clusters")

    # build legend manually using Line2D proxies so legend markers match cluster colors
    handles = []
    labels = []
    for lab in unique_labels:
        color = cmap(norm(lab))
        # create a proxy handle with markerfacecolor set to the cluster color
        handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=7))
        labels.append(f"cluster {int(lab)}")
    ax.legend(handles, labels, title="Clusters", bbox_to_anchor=(1.01, 1), loc="upper left")

    # add a colorbar that shows discrete cluster labels (ScalarMappable used to drive the colorbar)
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])  # required by colorbar API even if empty
    cbar = fig.colorbar(mappable, ax=ax, ticks=unique_labels)
    cbar.set_label("cluster label")

    plt.tight_layout()
    return fig