import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp

def classification_probability(
    x: float,
    samples: np.ndarray,
    name: str = None,
    sigma_logT90: float = 0.0,
    model: str = 'short'
):
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
    
    plt.figure()
    if model in ('short', 'both'):
        mean_h     = np.mean(h)
        median_h   = np.median(h)
        p5_h       = np.percentile(h,5)
        p95_h      = np.percentile(h,95)

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

        plt.hist(g, bins=bins, density=True, alpha=0.55, edgecolor='k')
        plt.axvline(mean_g, color='blue', linestyle='-', label=f'mean={mean_g:.3f}', linewidth = 3)
        plt.axvline(median_g, color='green', linestyle='--', label=f'median={median_g:.3f}', linewidth = 3)
        plt.axvline(p5_g, color='darkgrey', linestyle=':', label=f'5%={p5_g:.3f}', linewidth = 2.5)
        plt.axvline(p95_g, color='darkgrey', linestyle=':', label=f'95%={p95_g:.3f}', linewidth = 2.5)    
    
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
        
    plt.savefig(filepath + '.png', dpi=300, bbox_inches='tight')
    plt.savefig(filepath + '.pdf', dpi=300, bbox_inches='tight')
    plt.show()
        
