"""
Collapsed Gibbs sampler for a Dirichlet Process Gaussian mixture
with a Normal-Inverse-Gamma prior on (mu, sigma^2) for each component.

This code keeps the collapsed Gibbs machinery for sampling cluster
assignments (i.e. integrating out mu and sigma^2 when assigning points).
"""

from collections import Counter
import numpy as np
from scipy import stats


class SuffStat:
    """Sufficient statistics stored per cluster: N, sum_x, sum_x2.
    mean and S (sum of squared deviations) are computed on demand.
    """
     # Avoids per-instance __dict__: fixes attribute names and saves memory
    __slots__ = ("N", "sum_x", "sum_x2")

    def __init__(self, N=0, sum_x=0.0, sum_x2=0.0):
        self.N      = int(N)
        self.sum_x  = float(sum_x)
        self.sum_x2 = float(sum_x2)

    def copy(self):
        return SuffStat(self.N, self.sum_x, self.sum_x2)

    @property
    def mean(self):
        return self.sum_x / self.N if self.N > 0 else 0.0

    @property
    def S(self):
        # sum_{i}(x_i - mean)^2 = sum_x2 - N*mean^2
        if self.N <= 0:
            return 0.0
        m = self.mean
        return self.sum_x2 - self.N * (m * m)

    def add(self, x):
        self.N      += 1
        self.sum_x  += x
        self.sum_x2 += x * x

    def remove(self, x):
        if self.N <= 0:
            raise ValueError("Removing from empty suffstat")
        # subtract and guard when N becomes zero
        self.N      -= 1
        self.sum_x  -= x
        self.sum_x2 -= x * x
        if self.N == 0:
            self.sum_x = 0.0
            self.sum_x2 = 0.0


class CollapsedGibbsDP:
    """Collapsed Gibbs sampler for DP Gaussian mixture with NIG prior.

    Prior (component-level):
      mu | sigma2 ~ Normal(mu0, sigma2 / kappa0)
      sigma2      ~ Inv-Gamma(alpha0, beta0)

    Predictive for a new x given cluster data is a Student-t with
    location mu_n, scale sqrt( beta_n*(kappa_n+1)/(alpha_n*kappa_n) ),
    and degrees of freedom nu = 2*alpha_n, where the "_n" parameters are
    posterior hyperparameters after observing the cluster points.

    Parameters
    ----------
    data : array-like
        1D data points.
    alpha : float
        DP concentration parameter.
    mu0, kappa0, alpha0, beta0 : floats
        NIG prior hyperparameters.
    rng_seed : int
        Random generator
    init_clusters : int or None
        If None (default), start with one cluster that contains all points (legacy behavior).
        If int k >= 1, start with k clusters and assign each data point randomly to one of them.
        If k > n_points it will be clamped to n_points.
    """

    def __init__(self, data, alpha=1.0, mu0=0.0, kappa0=1e-6, alpha0=1e-3, beta0=1e-3, init_clusters=None, rng_gen=None):
        if rng_gen is None:
            self.rng    = np.random.default_rng(1234)
        else: self.rng = rng_gen
        self.data_  = np.asarray(data, dtype=float)
        self.alpha_ = float(alpha)

        # NIG prior hyperparameters
        if kappa0<=0 or alpha0<=0 or beta0<=0:
            raise ValueError("kappa0, alpha0 and beta0 must be > 0")
        self.mu0    = float(mu0)
        self.kappa0 = float(kappa0)  # should be > 0
        self.alpha0 = float(alpha0)  # > 0 in practice
        self.beta0  = float(beta0)   # > 0 in practice

        # initial state
        self.cluster_ids_ = []
        self.suffstats    = {}  # dict cluster_id -> SuffStat
        self.assignment   = []

        # start assignments (optionally with a given number of clusters)
        self._init_assignments(init_clusters)

    def _init_assignments(self, init_clusters=None):
        """Initialize cluster assignments.

        If init_clusters is None: one cluster with all points.
        If init_clusters is int k >= 1: create k clusters and assign each point
        uniformly at random to one of them (using self.rng).
        """
        n = self.data_.size
        if n == 0:
            return

        if init_clusters is None:
            # start with 1 cluster containing all points
            self.cluster_ids_ = [0]
            ss = SuffStat()
            for x in self.data_:
                ss.add(x)
            self.suffstats = {0: ss}
            self.assignment = [0 for _ in range(n)]
            return

        # initialize with a specified number of clusters, randomly assigning points
        k = int(init_clusters)
        if k < 1:
            k = 1
        # cannot have more clusters than data points meaningfully
        k = min(k, n)

        # create cluster ids 0..k-1
        self.cluster_ids_ = list(range(k))
        # initialize suffstats for each cluster
        self.suffstats = {cid: SuffStat() for cid in self.cluster_ids_}

        # random assignments in {0,...,k-1}
        # If number of clusters equals number of points, assign a permutation so every
        # point gets its own distinct cluster. Otherwise assign uniformly at random.
        if k == n:
            assigns = self.rng.permutation(k)
        else:
            assigns = self.rng.integers(0, k, size=n)
        self.assignment = [int(a) for a in assigns]

        # populate suffstats according to assignments
        for idx, x in enumerate(self.data_):
            cid = self.assignment[idx]
            self.suffstats[cid].add(float(x))

    # ---------- posterior / predictive helpers ----------
    def _posterior_hyperparams(self, ss: SuffStat):
        """Return (mu_n, kappa_n, alpha_n, beta_n) posterior hyperparams given suffstat ss."""
        N = ss.N
        if N == 0:
            return self.mu0, self.kappa0, self.alpha0, self.beta0
        mean = ss.mean
        S = ss.S
        kappa_n = self.kappa0 + N
        mu_n = (self.kappa0 * self.mu0 + N * mean) / kappa_n
        alpha_n = self.alpha0 + 0.5 * N
        beta_n = self.beta0 + 0.5 * S + (self.kappa0 * N * (mean - self.mu0) ** 2) / (2.0 * kappa_n)
        return mu_n, kappa_n, alpha_n, beta_n

    def _log_predictive_t(self, ss: SuffStat, x):
        """Log predictive density for x given cluster suffstat ss (Student-t).

        If ss.N == 0 this is the prior predictive (also Student-t).
        """
        mu_n, kappa_n, alpha_n, beta_n = self._posterior_hyperparams(ss)
        # degrees of freedom
        nu = 2.0 * alpha_n
        # scale (std) for Student-t
        scale2 = beta_n * (kappa_n + 1.0) / (alpha_n * kappa_n)
        scale = np.sqrt(max(scale2, 1e-16))
        # use scipy.stats.t for Student-t logpdf
        return stats.t.logpdf(x, df=nu, loc=mu_n, scale=scale)

    # ---------- cluster helpers ----------
    def _create_cluster(self):
        new_id = max(self.cluster_ids_) + 1 if len(self.cluster_ids_) > 0 else 0
        self.cluster_ids_.append(new_id)
        self.suffstats[new_id] = SuffStat()
        return new_id

    def _destroy_cluster(self, cid):
        # remove only if empty
        if self.suffstats[cid].N != 0:
            raise RuntimeError("Attempt to destroy non-empty cluster")
        del self.suffstats[cid]
        self.cluster_ids_.remove(cid)

    # ---------- assignment sampling (collapsed) ----------
    def _log_cluster_assignment_score(self, cid):
        if cid == "new":
            return np.log(self.alpha_)
        else:
            return np.log(self.suffstats[cid].N)

    def _cluster_assignment_logprobs(self, data_id):
        x = float(self.data_[data_id])
        # consider all existing clusters and a "new" cluster
        labels = list(self.cluster_ids_) + ["new"]
        logps = np.zeros(len(labels), dtype=float)
        for i, cid in enumerate(labels):
            if cid == "new":
                ss = SuffStat()  # empty suffstat -> prior predictive
            else:
                ss = self.suffstats[cid]
            lp = self._log_predictive_t(ss, x)
            lp += self._log_cluster_assignment_score(cid)
            logps[i] = lp
        # numerically stable normalization: subtract max
        m = logps.max()
        w = np.exp(logps - m)
        probs = w / w.sum()
        return labels, probs

    def _sample_assignment_for(self, data_id):
        labels, probs = self._cluster_assignment_logprobs(data_id)
        # draw
        choice = self.rng.choice(len(labels), p=probs)
        chosen = labels[choice]
        if chosen == "new":
            chosen = self._create_cluster()
        return int(chosen)

    # ---------- Collapsed Gibbs Sampling ----------
    def gibbs_step(self):
        # iterate through datapoints 
        # Randomly permute the data indices each iteration 
        # to avoid order bias and improve mixing.
        order = self.rng.permutation(self.data_.size)
        for data_id in order:
            x = self.data_[data_id]
            # current cluster
            old_cid = self.assignment[data_id]
            
            # remove point from old cluster
            # suffstats = dict {cluster_id -> SuffStat}
            # 'remove' is a function of SuffStat class:
            # removes x and updates sufficient statistics for cluster 'old_cid' 
            self.suffstats[old_cid].remove(x)
            
            # if cluster emptied, prune it
            if self.suffstats[old_cid].N == 0:
                # destroy immediately to avoid using it as existing cluster
                self._destroy_cluster(old_cid)

            # sample a new cluster (creates new cluster if needed)
            # calculates p(zi=k|z¬i,x,α) for each cluster k and randomly extracts one,
            # weighted by the calculated probabilities
            new_cid = self._sample_assignment_for(data_id)
            
            # add datapoint to the chosen suffstat
            # 'add' is a function of SuffStat class:
            # adds x and updates sufficient statistics for cluster 'new_cid' 
            self.suffstats[new_cid].add(x)
            
            # record assignment (updates zi)
            self.assignment[data_id] = new_cid

    def run(self, n_iters=100, verbose=False):
        """Run the Gibbs sampler.

        Parameters
        ----------
        n_iters : int
            Number of Gibbs iterations.
        verbose : bool
            Print progress.
        """
        for it in range(n_iters):
            self.gibbs_step()
            # print every 10%
            if verbose and (it + 1) % max(1, n_iters // 10) == 0:
                counts = Counter(self.assignment)
                print(f"Iter {it+1}/{n_iters}   num_clusters={len(self.cluster_ids_)}   counts={dict(counts)}")

    def get_state(self):
        return {
            'cluster_ids_': list(self.cluster_ids_),
            'assignment': list(self.assignment),
            'suffstats': {cid: self.suffstats[cid].copy() for cid in self.cluster_ids_},
            'alpha_': self.alpha_,
            'mu0': self.mu0,
            'kappa0': self.kappa0,
            'alpha0': self.alpha0,
            'beta0': self.beta0,
        }
