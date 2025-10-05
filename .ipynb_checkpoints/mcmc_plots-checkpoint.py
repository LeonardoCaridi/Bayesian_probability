import os
from typing import Callable, List, Union, Optional, Sequence, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde

from mcmc import autocorrelation, next_pow_two, log_T90_distribution

# Default parameter labels
DEFAULT_PARAM_LABELS = [
    'w',
    r'$\mu_1$',
    r'$\delta = \mu_2 - \mu_1$',
    r'$\sigma_1$',
    r'$\sigma_2$',
]


def _ensure_outdir(outdir: str) -> None:
    """Create output directory if it does not exist."""
    os.makedirs(outdir, exist_ok=True)


def plot_trace_mcmc(
    samples: np.ndarray,
    filename: str,
    theta_true: Optional[np.ndarray] = None,
    param_labels: Optional[Sequence[str]] = None,
    burnin: Optional[int] = 3000,
    outdir: str = "figure",
    save: bool = True,
    figsize: Tuple[float, float] = (8, 8),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot trace (iteration vs. sampled value) for each parameter in ``samples``.

    Parameters
    ----------
    samples : np.ndarray
        2D array with shape (n_iterations, n_parameters). Each column is a chain for a parameter.
    filename: str
        Name of the file
    theta_true : np.ndarray
        1D array with the "true" or reference values for each parameter. Length must equal n_parameters.
    param_labels : sequence of str, optional
        Labels for parameters plotted on the y-axis. If None, DEFAULT_PARAM_LABELS is used.
    burnin : int or None, optional
        If int, a vertical dotted line is drawn at this iteration to indicate a burn-in/cut-off.
        If None, no vertical burn-in line is drawn. Default: 3000.
    outdir : str, optional
        Directory where PNG/PDF files are saved when ``save`` is True. Default: 'figure'.
    save : bool, optional
        If True, save the figure as both PNG and PDF in ``outdir``. Default True.
    figsize : tuple, optional
        Figure size in inches. Default (8, 8).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    axes : numpy.ndarray
        Array of Axes objects (one per parameter).
    """
    # Basic validation
    if samples.ndim != 2:
        raise ValueError("`samples` must be a 2D array with shape (n_iters, n_params)")
    n_iters, n_params = samples.shape
    if theta_true is not None and theta_true.shape[0] != n_params:
        raise ValueError("Length of `theta_true` must match number of columns in `samples`")

    labels = list(param_labels) if param_labels is not None else DEFAULT_PARAM_LABELS[:n_params]

    # ----- plotting defaults -----
    marker_size = 4.5
    line_width = 1.2
    hline_color = '#d62728'
    vline_color = '#2b2b2b'
    hline_width = 2.2
    vline_width = 2.0

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0.15, 0.85, n_params)]

    fig, axes = plt.subplots(nrows=n_params, ncols=1, sharex=True, figsize=figsize, dpi=200)
    # If only one parameter, axes is not an array; force it to be an array for uniform handling
    if n_params == 1:
        axes = np.array([axes])

    fig.suptitle("Trace plots MCMC", fontsize=16, fontweight='bold', y=0.97)

    for i, ax in enumerate(axes):
        ax.plot(
            samples[:, i],
            marker='o',
            markersize=marker_size,
            markeredgewidth=0.005,
            markeredgecolor='k',
            linestyle='-',
            linewidth=line_width,
            alpha=0.65,
            color=colors[i],
        )

        # Horizontal line: true/reference value
        if theta_true is not None:
            ax.axhline(theta_true[i], color=hline_color, linestyle='--', linewidth=hline_width)

        # Vertical line: burn-in cut-off
        if burnin is not None:
            ax.axvline(burnin, color=vline_color, linestyle=':', linewidth=vline_width)

        # Labels and grid
        ax.set_ylabel(labels[i], fontsize=15)
        ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
        if i < n_params - 1:
            plt.setp(ax.get_xticklabels(), visible=False)

    axes[-1].set_xlabel("Iteration", fontsize=15)

    # Legend for the reference lines
    line_handles = [
        Line2D([0], [0], color=hline_color, lw=hline_width, linestyle='--'),
        Line2D([0], [0], color=vline_color, lw=vline_width, linestyle=':'),
    ]
    line_labels = ['True value', 'Burn-in (cut-off)']
    fig.legend(line_handles, line_labels, loc='upper right', fontsize=11, frameon=True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _ensure_outdir(outdir)
        png_path = os.path.join(outdir, filename + '.png')
        pdf_path = os.path.join(outdir, filename + '.pdf')
        fig.savefig(png_path, bbox_inches='tight', dpi=300)
        fig.savefig(pdf_path, bbox_inches='tight', dpi=300)
        print(f"Saved: {png_path} (PNG 300 dpi) and {pdf_path} (PDF)")

    plt.show()
    return fig, axes


def plot_marginal_distributions(
    samples: np.ndarray,
    filename: str,
    theta_true: Optional[np.ndarray] = None,
    param_tuples: Optional[Sequence[Tuple[int, str]]] = None,
    outdir: str = "figure",
    save: bool = True,
    figsize: Tuple[float, float] = (13.33, 3.8),
) -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Create marginal posterior histograms.

    Parameters
    ----------
    samples : np.ndarray
        2D array (n_samples, n_parameters) with posterior draws.
    filename: str
        Name of the file
    theta_true : np.ndarray
        Reference values for each parameter; used to draw a vertical line.
    param_tuples : sequence of (index, label), optional
        If provided, this controls which columns of samples are plotted and their labels.
        If None, the first five parameters are used and labelled with defaults.
        Example: [(0, 'w'), (1, r'$\\mu_1$'), ...]
    outdir : str, optional
        Directory where PNG/PDF files are saved when ``save`` is True.
    save : bool, optional
        If True, save the figure. Default True.
    figsize : tuple, optional
        Figure size in inches. Default tuned for 16:9 slide width.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of Axes
    """
    # Choose default parameters if none provided
    if param_tuples is None:
        # use up to the first 5 default labels
        param_tuples = [(i, DEFAULT_PARAM_LABELS[i]) for i in range(min(5, samples.shape[1]))]

    n_panels = len(param_tuples)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0.15, 0.85, n_panels)]

    # Improve default rcParams for presentation
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'figure.titlesize': 20,
    })
    plt.style.use('seaborn-v0_8-whitegrid')

    fig = plt.figure(2, figsize=figsize)
    gs = GridSpec(1, n_panels, figure=fig, wspace=0.26, hspace=0.16)

    axes = []
    for col, ((idx, title), color) in enumerate(zip(param_tuples, colors)):
        ax = fig.add_subplot(gs[0, col])
        samples_dist = samples[:, idx]

        ax.set_axisbelow(True)

        # Histogram
        n_bins = 30
        ax.hist(samples_dist, bins=n_bins, density=True,
                alpha=0.65, facecolor=color, edgecolor='k', linewidth=0.25, zorder=1)

        # True value + quantiles
        true_val = theta_true[idx]
        if col == 0:
            ax.axvline(true_val, color='red', linestyle='-', linewidth=2.0, zorder=6, label='true value')
        else:
            ax.axvline(true_val, color='red', linestyle='-', linewidth=2.0, zorder=6)

        q5, q50, q95 = np.percentile(samples_dist, [5, 50, 95])
        if col == 0:
            ax.axvline(q50, color='k', linestyle='--', linewidth=1.6, zorder=6, label='median')
        else:
            ax.axvline(q50, color='k', linestyle='--', linewidth=1.6, zorder=6)

        ax.axvline(q5, color='k', linestyle=':', linewidth=1.5, zorder=5)
        ax.axvline(q95, color='k', linestyle=':', linewidth=1.5, zorder=5)

        # Title and ticks
        ax.set_title(title, fontsize=17, pad=8)
        if col == 0:
            ax.set_ylabel('density', fontsize=16)
        else:
            ax.set_yticklabels([])

        # Annotation box showing summaries
        txt = f"med={q50:.3g}\n5–95%=[{q5:.3g}, {q95:.3g}]"
        ax.text(0.98, 0.95, txt, transform=ax.transAxes,
                ha='right', va='top', fontsize=12,
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white', alpha=0.95, edgecolor='none'),
                zorder=20)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=14)

        axes.append(ax)

    # Title and optional legend
    fig.suptitle('Posterior samples: Marginal distributions', fontsize=20, y=0.98)
    plt.subplots_adjust(top=0.84, left=0.05, right=0.995, bottom=0.10)

    # If any handle/labels exist, place a legend
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels,
                   loc='upper center',
                   bbox_to_anchor=(0.85, 1.03),
                   ncol=3,
                   frameon=True,
                   fancybox=True,
                   framealpha=0.9,
                   fontsize=14)

    if save:
        _ensure_outdir(outdir)
        png_path = os.path.join(outdir, filename + '.png')
        pdf_path = os.path.join(outdir, filename + '.pdf')
        fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.06)
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight', pad_inches=0.06)
        print(f"Saved: {png_path} (PNG 300 dpi) and {pdf_path} (PDF)")

    plt.show()
    return fig, axes


def plot_autocorrelations(
    samples: np.ndarray,
    filename: str,
    autocorr_func: Optional[Callable[[np.ndarray, Optional[int]], np.ndarray]] = None,
    param_labels: Optional[Sequence[str]] = None,
    max_lag: int = 5000,
    outdir: str = "figure",
    save: bool = True,
    figsize: Tuple[float, float] = (8, 8),
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot autocorrelation functions (ACF) for each parameter in ``samples_new``.

    Parameters
    ----------
    samples: np.ndarray
        2D array with shape (n_samples, n_parameters) containing posterior draws.
    filename: str
        Name of the file
    autocorr_func : callable, optional
        Function to compute the autocorrelation of a 1D array. Signature must be
        f(x: np.ndarray) -> np.ndarray. If None, a default implementation is used.
    param_labels : sequence of str, optional
        Labels for each parameter. If None, DEFAULT_PARAM_LABELS is used (as many as needed).
    max_lag : int, optional
        Maximum lag to compute and plot (x-axis limit). Default 5000.
    outdir : str, optional
        Directory to save output files.
    save : bool, optional
        Whether to save PNG/PDF files.
    figsize : tuple, optional

    Returns
    -------
    fig, axes
        Matplotlib objects for further customization.
    """
    if samples.ndim != 2:
        raise ValueError("`samples` must be a 2D array (n_samples, n_params)")

    n_samples, n_params = samples.shape
    labels = list(param_labels) if param_labels is not None else DEFAULT_PARAM_LABELS[:n_params]

    if autocorr_func is None:
        autocorr_func = autocorrelation

    # compute acfs
    acfs = [autocorr_func(samples[:, i])[:max_lag] for i in range(n_params)]
    lags = [np.arange(len(a)) for a in acfs]

    # plotting params
    marker_size = 4.5
    line_width = 1.2

    fig, axes = plt.subplots(nrows=n_params, ncols=1, sharex=True, figsize=figsize, dpi=200)
    if n_params == 1:
        axes = np.array([axes])

    fig.suptitle("Autocorrelation functions", fontsize=16, fontweight='bold', y=0.92)

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0.15, 0.85, n_params)]

    global_ymin = min(a.min() for a in acfs)
    ymin = min(global_ymin, -0.05)
    ymax = 1.05

    for i, ax in enumerate(axes):
        ax.plot(lags[i], acfs[i], marker='o',
                markersize=marker_size, markeredgewidth=0.005,
                markeredgecolor='k', linestyle='-', linewidth=line_width,
                color=colors[i], alpha=0.65)
        ax.axhline(0, color='k', linestyle='--', linewidth=1.2)
        ax.set_ylabel(labels[i], fontsize=15)
        ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.7)
        ax.set_ylim(ymin, ymax)
        if i < len(axes) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        ax.set_xlim(0, max_lag)

    axes[-1].set_xlabel("Lag", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _ensure_outdir(outdir)
        png = os.path.join(outdir, filename + '.png')
        pdf = os.path.join(outdir, filename + '.pdf')
        fig.savefig(png, bbox_inches='tight', dpi=300)
        fig.savefig(pdf, bbox_inches='tight', dpi=300)

    plt.show()
    return fig, axes


def plot_distribution(
    logT90: np.ndarray,
    samples: np.ndarray,
    filename: str,
    sigma_logT90: Union[float, np.ndarray] = 0.0,
    theta_true: Optional[np.ndarray] = None,
    percentiles: Sequence[int] = (5, 50, 95),
    outdir: str = "figure",
    save: bool = True,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot the distribution of `logT90`.
    
    This function builds the predicted density at each `logT90` value for every posterior
    draw in `samples` by calling `log_T90_distribution(logT90, theta, sigma_logT90)` for each
    posterior sample `theta`. It then computes the requested percentiles (by default
    5th, 50th, 95th) across the posterior predictive ensemble and plots:
    - a background histogram of the observed `logT90` values (density=True)
    - the median predicted density as a black line
    - a shaded band between the lower and upper percentiles
    - (optionally) the "real" distribution computed at `theta_true` as a red dashed line
    
    
    Parameters
    ----------
    logT90 : np.ndarray
        1D array of x-values where the model density is evaluated (e.g. observed logT90 values).
    samples : np.ndarray
        2D array with shape (n_samples, n_params) containing posterior draws for model parameters.
    filename: str
        Name of the file
    sigma_logT90: float or np.ndarray
        Gaussian error for logT90. If 'float' constant for all data. Default 0.0
    theta_true : np.ndarray, optional
        Optional parameter vector for the "true" or simulated parameter values. If provided,
        the resulting density `exp(log_T90_distribution(logT90, theta_true))` will be plotted
        as the red dashed "Real" curve.
    percentiles : sequence of int, optional
        Percentiles to compute across posterior predictive models (default: (5, 50, 95)).
    outdir : str, optional
        Directory where PNG/PDF files will be saved if `save` is True.
    save : bool, optional
        Whether to save the figure files (PNG and PDF). Default True.
    figsize : tuple, optional
        Figure size in inches. Default (8, 5).
    
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure object.
    ax : matplotlib.axes.Axes
        The single Axes instance containing the plot.
    """
    # Input validation
    logT90 = np.asarray(logT90)
    if samples.ndim != 2:
        raise ValueError("`samples` must be a 2D array (n_samples, n_params)")

    # 1) build distribution models: stack exp(log_density) for each posterior sample
    distribution_models = np.vstack([
        np.exp(log_T90_distribution(logT90, samples[i, :], sigma_logT90))
        for i in range(samples.shape[0])
    ]) # shape -> (n_samples, n_points)
    
    # 2) compute percentiles along the posterior axis (axis=0)
    p_low, p_med, p_high = np.percentile(distribution_models, percentiles, axis=0)
    
    # 3) sort x and corresponding vectors for smooth plotting
    order = np.argsort(logT90)
    x_sorted = logT90[order]
    m_sorted = p_med[order]
    l_sorted = p_low[order]
    h_sorted = p_high[order]

    # 4) compute the real distribution if theta_true provided
    if theta_true is not None:
        real = np.exp(log_T90_distribution(logT90, theta_true, sigma_logT90))
        real_sorted = real[order]
    
    # 5) plotting
    fig, ax = plt.subplots(figsize=figsize)
    
    # background histogram of observations
    ax.hist(logT90, bins=30, density=True, alpha=0.35,
    facecolor='lightgrey', edgecolor='none', label='logT90 histogram', zorder=0)
    
    # shaded 5-95% band
    ax.fill_between(x_sorted, l_sorted, h_sorted, alpha=0.4,
    facecolor='turquoise', label=f'{percentiles[0]}-{percentiles[-1]} percentile', zorder=1)
    
    # real distribution (if available)
    if theta_true is not None:
        ax.plot(x_sorted, real_sorted, 'r--', label='Real', zorder=2)
    
    # median
    ax.plot(x_sorted, m_sorted, 'k-', label=f'Median ({percentiles[1]}%)', zorder=2)
     
    ax.set_xlabel('logT90')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    
    if save:
        _ensure_outdir(outdir)
        png = os.path.join(outdir, filename + '.png')
        pdf = os.path.join(outdir, filename + '.pdf')
        fig.savefig(png, bbox_inches='tight', dpi=300)
        fig.savefig(pdf, bbox_inches='tight', dpi=300)
    
    plt.show()
    return fig, ax


def pairplot(
    samples: np.ndarray,
    filename: str,
    theta_true: Optional[np.ndarray] = None,
    param_tuples: Optional[Sequence[Tuple[int, str]]] = None,
    figsize: Tuple[float, float] = (10, 10),
    bins: str = 30,
    scatter_alpha: float = 0.35,
    marker_size: int = 8,
    show_corr: bool = True,
    outdir: str = "figure",
    save: bool = True,
    contour_levels: Sequence[float] = (0.5, 0.9),
    contour_linestyles: Sequence[str] = ('--', ':'),
    kde_gridsize: int = 100,
) -> Tuple[plt.Figure, Dict[Tuple[int,int], plt.Axes]]:
    """
    Parametri
    ---------
    samples : np.ndarray
        Array (n_samples, n_params) con i campioni MCMC.
    filename : str
        Name of the file
    theta_true : np.ndarray
        Reference values for each parameter; used to draw a vertical line.
    param_tuples : sequence of (index, label), optional
        If provided, this controls which columns of samples are plotted and their labels.
        If None, the first five parameters are used and labelled with defaults.
        Example: [(0, 'w'), (1, r'$\\mu_1$'), ...]
    figsize : tuple, optional
        Figure size in inches. Default (10,10).
    bins : str or int
        Number of bins in histograms.
    scatter_alpha, marker_size : float, int
    show_corr : bool
        Print the correlation coefficient in each axes.
    outdir : str, optional
        Directory where PNG/PDF files will be saved if `save` is True.
    save : bool, optional
        Whether to save the figure files (PNG and PDF). Default True.

    Ritorna
    -------
    (fig, axes)
        fig: oggetto matplotlib.figure.Figure
        axes: dict con chiavi (i,j) che mappano alle Axes usate (solo per i>=j)
    """

    samples = np.asarray(samples)
    n_samples, n_params = samples.shape

    # Choose default parameters if none provided
    if param_tuples is None:
        # use up to the first 5 default labels
        param_tuples = [(i, DEFAULT_PARAM_LABELS[i]) for i in range(min(5, samples.shape[1]))]

    k = len(param_tuples)

    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0.15, 0.85, k)]

    # styling
    plt.style.use('seaborn-v0_8-whitegrid')
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(k, k, figure=fig, wspace=0.06, hspace=0.06)

    axes = {}

    for i in range(k):
        idx_i, label_i = param_tuples[i]
        xi = samples[:, idx_i]

        for j in range(0, i + 1):  # include diagonal
            idx_j, label_j = param_tuples[j]
            ax = fig.add_subplot(gs[i, j])
            axes[(i, j)] = ax

            # diagonal: histogram
            if i == j:
                # histogram density
                ax.hist(xi, bins=bins, density=True, alpha=0.65, facecolor=colors[i],
                        edgecolor='k', linewidth=0.25)
                p5, p50, p95 = np.percentile(xi, [5,50,95])
                ax.axvline(p5, color='k', linestyle=':', linewidth=1.2)
                ax.axvline(p50, color='k', linestyle='--', linewidth=1.6)
                ax.axvline(p95, color='k', linestyle=':', linewidth=1.2)

                # true value vertical line
                if theta_true is not None:
                    try:
                        tv = float(theta_true[idx_i])
                        ax.axvline(tv, color='red', linestyle='-', linewidth=1.6)
                    except Exception:
                        pass

                if i < k - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(label_i)

                if j != 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel('w')

            # off-diagonal lower triangle: scatter 
            else:
                xj = samples[:, idx_j]
                ax.scatter(xj, xi, s=marker_size, alpha=scatter_alpha, linewidth=0)

                # Try to compute 2D KDE and contours for specified probability levels
                try:
                    # prepare grid over the data range with small padding
                    xmin, xmax = np.min(xj), np.max(xj)
                    ymin, ymax = np.min(xi), np.max(xi)
                    xpad = 0.05 * (xmax - xmin) if xmax > xmin else 0.1
                    ypad = 0.05 * (ymax - ymin) if ymax > ymin else 0.1
                    x_min, x_max = xmin - xpad, xmax + xpad
                    y_min, y_max = ymin - ypad, ymax + ypad

                    xi_lin = np.linspace(x_min, x_max, kde_gridsize)
                    yi_lin = np.linspace(y_min, y_max, kde_gridsize)
                    X, Y = np.meshgrid(xi_lin, yi_lin)

                    # evaluate kde
                    values = np.vstack([xj, xi])
                    kde = gaussian_kde(values)
                    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

                    # compute contour thresholds that enclose given probability mass
                    dx = (xi_lin[1] - xi_lin[0]) if kde_gridsize > 1 else 1.0
                    dy = (yi_lin[1] - yi_lin[0]) if kde_gridsize > 1 else 1.0
                    area = dx * dy

                    z_flat = Z.ravel()
                    # sort in descending order
                    idx_sort = np.argsort(z_flat)[::-1]
                    z_sorted = z_flat[idx_sort]
                    cumsum = np.cumsum(z_sorted) * area  # cumulative probability as we include high-density cells

                    thresholds = []
                    for p in contour_levels:
                        # find smallest density threshold so that cumulative >= p
                        try:
                            pos = np.searchsorted(cumsum, p)
                            # If pos is beyond array, use min density
                            if pos >= len(z_sorted):
                                thresh = z_sorted[-1]
                            else:
                                thresh = z_sorted[pos]
                        except Exception:
                            thresh = np.percentile(z_flat, 100*(1-p))
                        thresholds.append(thresh)

                    # draw contours: order them so that larger-prob (0.9) is inner; but levels should be increasing
                    # matplotlib's contour expects levels in increasing order; provide them sorted
                    # we want e.g. outer contour for 0.5 (lower density threshold) and inner for 0.9 (higher threshold)
                    # sort thresholds and corresponding linestyles
                    sorted_pairs = sorted(zip(thresholds, contour_levels, contour_linestyles), key=lambda t: t[0])
                    level_vals = [p[0] for p in sorted_pairs]
                    linestyles_sorted = [p[2] for p in sorted_pairs]

                    # plot contours: use black lines and chosen linestyles
                    cs = ax.contour(X, Y, Z, levels=level_vals, linewidths=1.2, colors='k')
                    # set linestyles to match sorted order
                    for cidx, coll in enumerate(cs.collections):
                        coll.set_linestyle(linestyles_sorted[cidx])

                    # optionally label contours with their probability (small text)
                    # we avoid clabel to keep plot clean; instead we can add legend entries later if desired

                except Exception:
                    # if KDE fails (e.g. singular matrix with identical points), just skip contours
                    pass

                # true value vertical line an horizontal line
                if theta_true is not None:
                    try:
                        th = float(theta_true[idx_i])
                        tv = float(theta_true[idx_j])
                        ax.axvline(tv, color='red', linestyle='-', linewidth=1.6)
                        ax.axhline(th, color='red', linestyle='-', linewidth=1.6)
                    except Exception:
                        pass
                        
                # correlation
                if show_corr:
                    try:
                        r = np.corrcoef(xj, xi)[0, 1]
                        ax.text(0.02, 0.92, f"r={r:.2f}", transform=ax.transAxes,
                                fontsize=15, va='top')
                    except Exception:
                        pass

                # tidy ticks: only bottom row has xlabels, only first col has ylabels
                if i < k - 1:
                    ax.set_xticklabels([])
                else:
                    ax.set_xlabel(label_j)

                    # If this is the bottom-left panel force only two xticks
                    if j == 0:
                        ax.set_xticks([0.275,0.325])
                    if j == 3:
                        ax.set_xticks([0.8,0.9])

                if j != 0:
                    ax.set_yticklabels([])
                else:
                    ax.set_ylabel(label_i)

            # remove top/right spines for a cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)


    # global title and layout
    fig.suptitle('Pairplot', fontsize=16, y=0.95)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        _ensure_outdir(outdir)
        png = os.path.join(outdir, filename + '.png')
        pdf = os.path.join(outdir, filename + '.pdf')
        fig.savefig(png, bbox_inches='tight', dpi=300)
        fig.savefig(pdf, bbox_inches='tight', dpi=300)

    return fig, axes

