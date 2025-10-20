import numpy as np
import pandas as pd 
from scipy.special import logsumexp
from scipy.stats import beta
from dynesty import NestedSampler


class MixtureNestedSampler:
    """Class to run nested sampling on Gaussian mixtures (H0: 2 components, H1: 3 components).

    Required external parameters:
      - logT90: array-like of data 
      - sigma_logT90: scalar or array of gaussian errors on logT90
    
    Optional: prior hyperparameters and sampler settings.
    """

    def __init__(
        self,
        logT90,
        sigma_logT90 = 0.0,
        beta_a       = 2.0,
        beta_b       = 2.0,
    ):
        # data
        self.logT90       = np.asarray(logT90)
        self.sigma_logT90 = np.asarray(sigma_logT90)
            
        # prior hyperparams stick-breaking
        self.beta_a = beta_a
        self.beta_b = beta_b

        # data-adaptive priors (
        xmin, xmax   = np.min(self.logT90), np.max(self.logT90)
        self.mu_low  = xmin 
        self.mu_high = xmax 
        self.sd_data = max(1e-3, np.std(self.logT90))

        # sigma bounds (in log scale), correspond to [0.1, 6] with real data
        sigma_min    = max(0.01, self.sd_data * 0.05)
        sigma_max    = max(0.1, self.sd_data * 3.0)
        self.lns_min = np.log(sigma_min)
        self.lns_max = np.log(sigma_max)
        
    # -------------------------
    # helper: log-density for a mixture with k components
    # -------------------------
    def logT90_distribution(self, weights, mus, sigmas, x=None, sigma_logT90=None):
        """Costruisce la distribuzione del logT90

        weights: array (k,)
        mus: array (k,)
        sigmas: array (k,) (sigma in scala lineare)
        """
        if x is None:
            x = self.logT90
        k = len(weights)
        var_components = np.asarray(sigmas) ** 2

        # safe broadcasting for sigma_logT90 (can be scalar or an array matching x) 
        if sigma_logT90 is None:
            sigma_logT90 = self.sigma_logT90
        sigma_err2 = np.asarray(sigma_logT90) ** 2

        logNs = []
        for i in range(k):
            # var is scalar or array (shape == x.shape)
            var = var_components[i] + sigma_err2
            # numerical check: avoid var <= 0
            if np.any(var <= 0):
                return -np.inf
            # log-density for component (elementwise)
            logN = -0.5 * (np.log(2 * np.pi * var) + ((x - mus[i]) ** 2) / var)
            logNs.append(np.log(weights[i]) + logN)

        stacked = np.vstack(logNs)  # shape (k, N)
        # per-datum log-probability (log-sum-exp across components), then sum over data
        lpd = logsumexp(stacked, axis=0)
        return lpd
        
    def log_likelihood_k(self, weights, mus, sigmas):
        """
        Calcola la log-likelihood totale per mixture con componenti gaussiane
        """
        return np.sum(self.logT90_distribution(weights, mus, sigmas))

    # -------------------------
    # H0: 2-component model
    # param: [v, mu1_raw, delta, ln_s1, ln_s2]
    # -------------------------
    def prior_transform_h0(self, u):
        """Transform from the unit cube to physical parameters for H0 (dimension 5).

        u: array-like of length 5 with values in (0,1)
        returns: array([v, mu1, delta, ln_s1, ln_s2])
        """
        v = beta.ppf(u[0], self.beta_a, self.beta_b)
        mu1 = self.mu_low + u[1] * (self.mu_high - self.mu_low)
        max_delta = max(1e-3, (self.mu_high - mu1))
        delta = u[2] * max_delta
        ln_s1 = self.lns_min + u[3] * (self.lns_max - self.lns_min)
        ln_s2 = self.lns_min + u[4] * (self.lns_max - self.lns_min)
        return np.array([v, mu1, delta, ln_s1, ln_s2])

    def loglik_h0(self, theta):
        """Log-likelihood (log-posterior up-to-constant) per H0 dato theta (5-dim)."""
        v, mu1, delta, ln_s1, ln_s2 = theta
        if not (0.0 < v < 1.0 and delta >= 0.0):
            return -np.inf
        w1   = v
        w2   = 1.0 - v
        mu2  = mu1 + delta
        sig1 = np.exp(ln_s1)
        sig2 = np.exp(ln_s2)
        if sig1 <= 0 or sig2 <= 0:
            return -np.inf
        return self.log_likelihood_k(np.array([w1, w2]), np.array([mu1, mu2]), np.array([sig1, sig2]))

    # ---------- H1: 3-component with delta parametrization ----------
    # param dim = 8
    def prior_transform_h1(self, u):
        """Transform from the unit cube to physical parameters for H1 (dimension 8).

        u: array-like of length 8
        returns: array([v1, v2, mu1, delta12, delta23, ln_s1, ln_s2, ln_s3])
        """
        v1  = beta.ppf(u[0], self.beta_a, self.beta_b)
        v2  = beta.ppf(u[1], self.beta_a, self.beta_b)
        mu1 = self.mu_low + u[2] * (self.mu_high - self.mu_low)

        max_d12 = max(1e-8, self.mu_high - mu1)
        delta12 = u[3] * max_d12
        mu2     = mu1 + delta12

        max_d23 = max(1e-8, self.mu_high - mu2)
        delta23 = u[4] * max_d23
        # mus: mu1, mu2, mu3

        ln_s1 = self.lns_min + u[5] * (self.lns_max - self.lns_min)
        ln_s2 = self.lns_min + u[6] * (self.lns_max - self.lns_min)
        ln_s3 = self.lns_min + u[7] * (self.lns_max - self.lns_min)

        return np.array([v1, v2, mu1, delta12, delta23, ln_s1, ln_s2, ln_s3])

    def loglik_h1(self, theta):
        v1, v2 = theta[0], theta[1]
        mu1, delta12, delta23 = theta[2], theta[3], theta[4]
        ln_s1, ln_s2, ln_s3 = theta[5], theta[6], theta[7]

        if not (0.0 < v1 < 1.0 and 0.0 < v2 < 1.0 and delta12 >= 0 and delta23 >= 0):
            return -np.inf

        w1   = v1
        w2   = (1.0 - v1) * v2
        w3   = (1.0 - v1) * (1.0 - v2)
        mu2  = mu1 + delta12
        mu3  = mu2 + delta23
        sigs = np.exp([ln_s1, ln_s2, ln_s3])
        if np.any(sigs <= 0.0):
            return -np.inf
        weights = np.array([w1, w2, w3])
        mus     = np.array([mu1, mu2, mu3])
        return self.log_likelihood_k(weights, mus, sigs)

    # ---------- function to perform N runs and collect logZ ----------
    def multirun_model(
        self,
        model_name,
        n_runs = 5,
        nlive  = 1200,
        dlogz  = 0.01, 
        bound  = 'multi',
        sample = 'rwalk',
        rng    = None):
        """Esegue n_runs di nested sampling per il modello scelto ('H0' o 'H1').

        Ritorna:
          - results: lista di dynesty.results
          - logZs: numpy array dei logZ finali per ogni run
        """
        if rng is None:
            rng = np.random.default_rng(1234)

        results = []
        logZs = []
        for i in range(n_runs):
            if model_name == 'H0':
                ndim     = 5
                loglik   = self.loglik_h0
                prior_tr = self.prior_transform_h0
            elif model_name == 'H1':
                ndim     = 8
                loglik   = self.loglik_h1
                prior_tr = self.prior_transform_h1
            else:
                raise ValueError("model_name must be 'H0' or 'H1'")

            sampler = NestedSampler(loglik, prior_tr, ndim=ndim, 
                                    nlive=nlive, bound=bound, sample=sample, rstate=rng)
            print(f"Run {i+1}/{n_runs} for {model_name}")
            sampler.run_nested(dlogz=dlogz, print_progress=True)
            res = sampler.results
            results.append(res)
            logZs.append(float(res.logz[-1]))
            print(f"  logZ = {logZs[-1]:.4f} ± {float(res.logzerr[-1]):.4f}")

        return results, np.array(logZs)

    # helper: run a single nested sampling and return the result
    def run_once(self, model_name, nlive=1200, dlogz=0.01, rng=None):
        if rng is None:
            rng = np.random.default_rng(1234)
        res_list, logZs = self.multirun_model(model_name, n_runs=1, nlive=nlive, dlogz=dlogz, rng=rng)
        return res_list[0], float(logZs[0])

    def extract_posterior(
        self,
        res,
        n_samples=2000, 
        rng=None):

        if rng is None:
            rng = np.random.default_rng(1234)
            
        # get samples in physical space and log-weights from dynesty result
        samples = np.asarray(res.samples)
        logw    = np.asarray(res.logwt)
        # normalize log-weights to get probabilities
        # logsumexp(logw) is equal to Z on the merged run, 
        # that do not correspond to the mean of the Z on each run (the difference is negligible)
        logw_norm = logw - logsumexp(logw)
        w = np.exp(logw_norm)

        # importance resampling to obtain posterior draws for quantiles
        idx = rng.choice(len(w), size=n_samples, replace=True, p=w)
        # post_samps
        return samples[idx]


    def params_propriety(self, post_samps, model, quantiles=(0.05, 0.16, 0.5, 0.84, 0.95)):
        """
        Extract posterior summaries from unweighted posterior samples.
    
        Parameters
        ----------
        post_samps : np.ndarray
            Posterior unweighted samples, shape (N, D)
        model_name : {'H0','H1'}
            Which model the result corresponds to (affects derived quantities)
        quantiles : tuple
            Quantiles to compute (default 5%,16%,50%,84%,95%).
    
        Returns
        -------
        out : dict
            Dictionary containing:
              - 'param_means': sample mean of parameters in physical space (array)
              - 'param_cov': sample covariance matrix (ndarray, shape (D,D))
              - 'quantiles': dict mapping parameter name -> quantile array
              - 'posterior_samples': ndarray (N, D) same as input
              - 'derived_means': dict of derived quantities means (component weights, mus, sigs)
              - 'derived_qs': dict of credible intervals for derived quantities
              - 'names': list of parameter names in order
        """
        post_samps = np.asarray(post_samps)
        if post_samps.ndim != 2:
            raise ValueError("post_samps must be a 2D array of shape (N, D)")
        n, D = post_samps.shape
    
        # parameter names per model
        if model == 'H0':
            names = ['v', 'mu1', 'delta', 'ln_s1', 'ln_s2']
        elif model == 'H1':
            names = ['v1', 'v2', 'mu1', 'delta12', 'delta23', 'ln_s1', 'ln_s2', 'ln_s3']
        else:
            raise ValueError("model_name must be 'H0' or 'H1'")
    
        if len(names) != D:
            raise ValueError(f"post_samps has {D} columns but model {model_name} expects {len(names)} parameters")
    
        # basic summaries on parameters (empirical, unweighted)
        param_means = np.mean(post_samps, axis=0)
        param_cov = np.cov(post_samps, rowvar=False)   # shape (D,D), unbiased (divides by N-1)
    
        # compute quantiles for each parameter from the posterior samples
        qs = {}
        pct = np.asarray(quantiles) * 100.0
        for i, name in enumerate(names):
            qs[name] = np.percentile(post_samps[:, i], pct)
    
        # derived quantities and their summaries (computed per-sample)
        if model == 'H0':
            v     = post_samps[:, 0]
            mu1   = post_samps[:, 1]
            delta = post_samps[:, 2]
            ln_s1 = post_samps[:, 3]
            ln_s2 = post_samps[:, 4]
    
            w1 = v
            w2 = 1.0 - v
            mu2 = mu1 + delta
            sig1 = np.exp(ln_s1)
            sig2 = np.exp(ln_s2)
    
            derived_means = {
                'weights': np.array([w1.mean(), w2.mean()]),
                'mus':    np.array([mu1.mean(), mu2.mean()]),
                'sigs':   np.array([sig1.mean(), sig2.mean()]),
            }
    
            derived_qs = {
                'w1': np.percentile(w1, pct),
                'w2': np.percentile(w2, pct),
                'mu1': np.percentile(mu1, pct),
                'mu2': np.percentile(mu2, pct),
                'sig1': np.percentile(sig1, pct),
                'sig2': np.percentile(sig2, pct),
            }
    
        else:  # H1
            v1    = post_samps[:, 0]
            v2    = post_samps[:, 1]
            mu1   = post_samps[:, 2]
            d12   = post_samps[:, 3]
            d23   = post_samps[:, 4]
            ln_s1 = post_samps[:, 5]
            ln_s2 = post_samps[:, 6]
            ln_s3 = post_samps[:, 7]
    
            # mixture weights for three components (in standard stick-breaking parametrization)
            w1 = v1
            w2 = (1.0 - v1) * v2
            w3 = (1.0 - v1) * (1.0 - v2)
    
            mu2 = mu1 + d12
            mu3 = mu2 + d23
            sig1 = np.exp(ln_s1)
            sig2 = np.exp(ln_s2)
            sig3 = np.exp(ln_s3)
    
            derived_means = {
                'weights': np.array([w1.mean(), w2.mean(), w3.mean()]),
                'mus':    np.array([mu1.mean(), mu2.mean(), mu3.mean()]),
                'sigs':   np.array([sig1.mean(), sig2.mean(), sig3.mean()]),
            }
    
            derived_qs = {
                'w1': np.percentile(w1, pct),
                'w2': np.percentile(w2, pct),
                'w3': np.percentile(w3, pct),
                'mu1': np.percentile(mu1, pct),
                'mu2': np.percentile(mu2, pct),
                'mu3': np.percentile(mu3, pct),
                'sig1': np.percentile(sig1, pct),
                'sig2': np.percentile(sig2, pct),
                'sig3': np.percentile(sig3, pct),
            }
    
        out = {
            'names': names,
            'param_means': param_means,
            'param_cov': param_cov,
            'quantiles': qs,
            'posterior_samples': post_samps,
            'derived_means': derived_means,
            'derived_qs': derived_qs,
        }
        return out

    def derived_params_propriety(self, post_samps, model, quantiles=(0.05, 0.16, 0.5, 0.84, 0.95)):
        info = self.params_propriety(post_samps, model, quantiles)

        if model == 'H0':
            mapping = {
                'weights': ('w1', 'w2'),
                'mus':     ('mu1', 'mu2'),
                'sigs':    ('sig1', 'sig2')
            }
            rows = []
            for param, (q_first, q_second) in mapping.items():
                for i, qname in enumerate((q_first, q_second)):
                    row = {'param': param, 'component': i, 'mean': info['derived_means'][param][i]}
                    for p, val in zip(quantiles, info['derived_qs'][qname]):
                        row[f'p{p}'] = val
                    rows.append(row)
        else: 
            mapping = {
                'weights': ('w1', 'w2', 'w3'),
                'mus':     ('mu1', 'mu2', 'mu3'),
                'sigs':    ('sig1', 'sig2', 'sig3')
            }
            rows = []
            for param, (q_first, q_second, q_third) in mapping.items():
                for i, qname in enumerate((q_first, q_second, q_third)):
                    row = {'param': param, 'component': i, 'mean': info['derived_means'][param][i]}
                    for p, val in zip(quantiles, info['derived_qs'][qname]):
                        row[f'p{p}'] = val
                    rows.append(row)
        
        
        df = pd.DataFrame(rows)
        # ordina per param e component
        df = df.sort_values(['param','component']).reset_index(drop=True)
        display(df)        
        return df
