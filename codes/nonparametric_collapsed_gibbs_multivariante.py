"""
Collapsed Gibbs sampler for a Dirichlet Process Gaussian mixture
with a Normal-Inverse-Wishart (NIW) prior on (mu, Sigma) for each component.

This file supports multivariate data (e.g. 2D points x = (x1,x2)).
It implements collapsed Gibbs for cluster assignments using the marginal
(predictive) Student-t distribution that results from integrating out (mu,Sigma)
under a NIW prior. It also provides functions to sample (mu, Sigma) from the
NIW posterior for a given cluster (augmentation / post-hoc draws).

At the top of the file we include the formulas used for the posterior and the
predictive Student-t (in comments), then the implementation follows.
"""

from collections import Counter
import numpy as np
from scipy import stats


# =====================
# Formule (NIW / predictive)
# =====================
# Notazione:
# - D: dimensione dei dati (qui D=2 nel tuo caso)
# - prior NIW( mu0, kappa0, nu0, Lambda0 )
#   mu | Sigma ~ N(mu0, Sigma / kappa0)
#   Sigma     ~ Inv-Wishart(nu0, Lambda0)
#
# Dati in un cluster: {x_i}_{i=1..N}, con sample mean x_bar e
# S = sum_i (x_i - x_bar)(x_i - x_bar)^T
# (si puo' calcolare anche come sum_xxT - N * x_bar x_bar^T)
#
# Posterior NIW (dati del cluster):
#   kappa_n = kappa0 + N
#   mu_n    = (kappa0 * mu0 + N * x_bar) / kappa_n
#   nu_n    = nu0 + N
#   Lambda_n = Lambda0 + S + (kappa0 * N / kappa_n) * (x_bar - mu0)(x_bar - mu0)^T
#
# Predictive marginal per un nuovo punto x (multivariate Student-t):
#   df = nu_n - D + 1
#   location = mu_n
#   scale matrix (Sigma_pred) = (kappa_n + 1) / (kappa_n * df) * Lambda_n
#
# La densita' multivariate Student-t a df gradi di liberta', media mu_n e scala Sigma_pred
# e':
#   p(x) = Gamma((df + D)/2) / ( Gamma(df/2) * (df*pi)^(D/2) * |Sigma_pred|^(1/2) )
#          * ( 1 + (1/df) (x - mu_n)^T Sigma_pred^{-1} (x - mu_n) )^{-(df + D)/2}
#
# Per campionare (mu, Sigma) dalla posteriore (NIW posterior):
#   - campiona Sigma ~ Inv-Wishart(nu_n, Lambda_n)
#   - campiona mu ~ N(mu_n, Sigma / kappa_n)
#
# Queste sono le relazioni implementate nel codice sottostante.


class SuffStat:
    """Sufficient statistics for multivariate data per cluster.

    Attributes
    ----------
    N : int
        Number of points.
    sum_x : ndarray, shape (D,)
        Sum of vectors x_i.
    sum_xxT : ndarray, shape (D,D)
        Sum of outer-products x_i x_i^T.
    """

    __slots__ = ("N", "sum_x", "sum_xxT")

    def __init__(self, D, N=0, sum_x=None, sum_xxT=None):
        self.N = int(N)
        self.sum_x = np.zeros(D) if sum_x is None else np.asarray(sum_x, dtype=float).copy()
        self.sum_xxT = np.zeros((D, D)) if sum_xxT is None else np.asarray(sum_xxT, dtype=float).copy()

    def copy(self):
        return SuffStat(self.sum_x.shape[0], self.N, self.sum_x.copy(), self.sum_xxT.copy())

    @property
    def mean(self):
        return self.sum_x / self.N if self.N > 0 else np.zeros_like(self.sum_x)

    @property
    def S(self):
        # S = sum_i (x_i - mean)(x_i - mean)^T = sum_xxT - N * mean mean^T
        if self.N <= 0:
            return np.zeros_like(self.sum_xxT)
        m = self.mean
        return self.sum_xxT - self.N * np.outer(m, m)

    def add(self, x):
        x = np.asarray(x, dtype=float)
        self.N += 1
        self.sum_x += x
        self.sum_xxT += np.outer(x, x)

    def remove(self, x):
        x = np.asarray(x, dtype=float)
        if self.N <= 0:
            raise ValueError("Removing from empty suffstat")
        self.N -= 1
        self.sum_x -= x
        self.sum_xxT -= np.outer(x, x)
        if self.N == 0:
            # reset to canonical empty
            self.sum_x[:] = 0.0
            self.sum_xxT[:, :] = 0.0


class CollapsedGibbsDP:
    """Collapsed Gibbs sampler for DP Gaussian mixture with NIW prior.

    Prior: mu | Sigma ~ N(mu0, Sigma / kappa0), Sigma ~ Inv-Wishart(nu0, Lambda0)

    rng_seed : seed for random initialization and internal RNG
    init_clusters : if None (default) -> single-cluster init;
                    if int k >= 1 -> start with k clusters and randomly assign points
    """

    def __init__(self, data, alpha=1.0, mu0=None, kappa0=1e-6, nu0=None, Lambda0=None, init_clusters=None, rng_seed=1234):
        self.rng = np.random.default_rng(rng_seed)
        self.data_ = np.asarray(data, dtype=float)
        if self.data_.ndim == 1:
            # treat as 1D column
            self.data_ = self.data_[:, None]
        self.N_total, self.D = self.data_.shape
        self.alpha_ = float(alpha)

        # default NIW hyperparams if not provided
        self.mu0 = np.zeros(self.D) if mu0 is None else np.asarray(mu0, dtype=float)
        self.kappa0 = float(kappa0)
        self.nu0 = float(self.D + 2) if nu0 is None else float(nu0)  # must be > D-1
        self.Lambda0 = np.eye(self.D) if Lambda0 is None else np.asarray(Lambda0, dtype=float)

        # initial state
        self.cluster_ids_ = []
        self.suffstats = {}  # dict cluster_id -> SuffStat
        self.assignment = []

        # traces for parameters
        self.param_traces_ = []
        self.counts_trace_ = []

        # start assignments (optionally with a given number of clusters)
        self._init_assignments(init_clusters)

    def _init_assignments(self, init_clusters=None):
        """Initialize cluster assignments.

        If init_clusters is None: one cluster with all points.
        If init_clusters is int k >= 1: create k clusters and assign each point
        uniformly at random to one of them (using self.rng).
        """
        n = self.N_total
        if n == 0:
            return

        if init_clusters is None:
            # start with 1 cluster containing all points
            self.cluster_ids_ = [0]
            ss = SuffStat(self.D)
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
        self.suffstats = {cid: SuffStat(self.D) for cid in self.cluster_ids_}

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
            self.suffstats[cid].add(x)

    # ---------- posterior / predictive helpers ----------
    def _posterior_hyperparams(self, ss: SuffStat):
        """Return (mu_n, kappa_n, nu_n, Lambda_n) for a cluster suffstat."""
        N = ss.N
        if N == 0:
            return self.mu0.copy(), self.kappa0, self.nu0, self.Lambda0.copy()
        mean = ss.mean
        S = ss.S
        kappa_n = self.kappa0 + N
        mu_n = (self.kappa0 * self.mu0 + N * mean) / kappa_n
        nu_n = self.nu0 + N
        diff = mean - self.mu0
        Lambda_n = self.Lambda0 + S + (self.kappa0 * N / kappa_n) * np.outer(diff, diff)
        return mu_n, kappa_n, nu_n, Lambda_n

    def _log_predictive_student_t(self, ss: SuffStat, x):
        """Log predictive density (multivariate Student-t) for point x given suffstat ss."""
        mu_n, kappa_n, nu_n, Lambda_n = self._posterior_hyperparams(ss)
        D = self.D
        df = nu_n - D + 1.0
        if df <= 0 or kappa_n <= 0:
            # fallback to broad multivariate normal
            cov = (self.Lambda0 + np.eye(D)) * 1e2
            return stats.multivariate_normal(self.mu0, cov).logpdf(x)
        # scale matrix for Student-t
        Sigma_pred = (kappa_n + 1.0) / (kappa_n * df) * Lambda_n
        # compute logpdf of multivariate Student-t manually
        x = np.asarray(x, dtype=float)
        xm = x - mu_n
        sign, logdet = np.linalg.slogdet(Sigma_pred)
        if sign <= 0:
            # numerical fallback
            Sigma_pred += np.eye(D) * 1e-8
            sign, logdet = np.linalg.slogdet(Sigma_pred)
        inv_S = np.linalg.inv(Sigma_pred)
        quad = float(xm.T.dot(inv_S).dot(xm))
        log_num = stats.gammaln((df + D) / 2.0)
        log_den = stats.gammaln(df / 2.0) + (D / 2.0) * np.log(df * np.pi) + 0.5 * logdet
        log_kernel = - (df + D) / 2.0 * np.log(1.0 + quad / df)
        return float(log_num - log_den + log_kernel)

    # ---------- cluster bookkeeping ----------
    def _create_cluster(self):
        new_id = max(self.cluster_ids_) + 1 if len(self.cluster_ids_) > 0 else 0
        self.cluster_ids_.append(new_id)
        self.suffstats[new_id] = SuffStat(self.D)
        return new_id

    def _destroy_cluster(self, cid):
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
        x = self.data_[data_id]
        labels = list(self.cluster_ids_) + ["new"]
        logps = np.zeros(len(labels), dtype=float)
        for i, cid in enumerate(labels):
            if cid == "new":
                ss = SuffStat(self.D)
            else:
                ss = self.suffstats[cid]
            lp = self._log_predictive_student_t(ss, x)
            lp += self._log_cluster_assignment_score(cid)
            logps[i] = lp
        m = logps.max()
        w = np.exp(logps - m)
        probs = w / w.sum()
        return labels, probs

    def _sample_assignment_for(self, data_id):
        labels, probs = self._cluster_assignment_logprobs(data_id)
        choice = self.rng.choice(len(labels), p=probs)
        chosen = labels[choice]
        if chosen == "new":
            chosen = self._create_cluster()
        return int(chosen)

    # ---------- Gibbs step ----------
    def gibbs_step(self):
        order = self.rng.permutation(self.N_total)
        for data_id in order:
            x = self.data_[data_id]
            old_cid = self.assignment[data_id]
            self.suffstats[old_cid].remove(x)
            if self.suffstats[old_cid].N == 0:
                self._destroy_cluster(old_cid)
            new_cid = self._sample_assignment_for(data_id)
            self.suffstats[new_cid].add(x)
            self.assignment[data_id] = new_cid

    def run(self, n_iters=100, verbose=False):
        for it in range(n_iters):
            self.gibbs_step()
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
            'nu0': self.nu0,
            'Lambda0': self.Lambda0,
        }


if __name__ == "__main__":
    # quick sanity check in 2D
    np.random.seed(0)
    x1 = np.random.multivariate_normal([-2.0, 0.0], np.diag([0.5**2, 0.3**2]), size=50)
    x2 = np.random.multivariate_normal([2.0, 1.0], np.diag([0.7**2, 0.6**2]), size=60)
    data = np.vstack([x1, x2])

    mu0 = np.zeros(2)
    Lambda0 = np.eye(2)
    sampler = CollapsedGibbsDP(data, alpha=1.0, mu0=mu0, kappa0=0.01, nu0=5.0, Lambda0=Lambda0)
    print("Initial clusters:", len(sampler.cluster_ids_))
    sampler.run(200, verbose=True, collect_param_samples=True, thin=5)
    st = sampler.get_state()
    print("Final num clusters:", len(st['cluster_ids_']))
    print("Counts:", Counter(st['assignment']))
