from collections import Counter, namedtuple
import numpy as np
from scipy import stats
import pandas as pd

# SuffStat: contains mean (theta), N and S = sum (x - mean)^2
SuffStat = namedtuple('SuffStat', 'theta N S')


class GibbsGaussianMixture:
    """Class encapsulating a Gibbs sampler for a Gaussian mixture model.

    Variant B (marginal on mu, sample tau then mu):
      - tau ~ Inv-Gamma((N-1)/2, S/2) when N >= 2
      - mu | tau ~ Normal(bar_x, tau / N)

    Main parameters:
      - data: 1D array of observations
      - num_clusters: number of mixture components
      - alpha: Dirichlet parameter for mixture weights
      - rng_seed: random seed (default 1234)
      - hyperparameters: dict with prior for empty clusters (mean, variance)
    """

    def __init__(self,
                 data=None,
                 num_clusters=3,
                 alpha=1.0,
                 rng_seed=1234,
                 hyperparameters=None,
                 init_means=None,
                 init_variance=0.01):
        self.rng = np.random.default_rng(rng_seed)
        self.data_ = np.asarray(data) if data is not None else np.array([])
        self.num_clusters_ = int(num_clusters)
        self.alpha_ = float(alpha)
        self.cluster_ids_ = list(range(self.num_clusters_))

        # hyperparameters used as fallback for empty clusters
        if hyperparameters is None:
            hyperparameters = {"mean": 0.0, "variance": 1.0}
        self.hyperparameters_ = hyperparameters

        # initial values
        if init_means is None:
            # default: spread means across [-1, 1]
            self.cluster_means = list(np.linspace(-1.0, 1.0, self.num_clusters_))
        else:
            self.cluster_means = list(init_means)

        # initialize cluster variances (tau = sigma^2)
        self.cluster_variances = [float(init_variance) for _ in self.cluster_ids_]

        # random initial assignments if data provided
        if self.data_.size > 0:
            self.assignment = list(self.rng.choice(self.cluster_ids_, size=self.data_.size))
        else:
            self.assignment = []

        # mixture weights (pi)
        self.pi = [self.alpha_ / self.num_clusters_ for _ in self.cluster_ids_]

        # placeholder for sufficient statistics
        self.suffstats = [None] * self.num_clusters_

        # compute initial sufficient statistics
        self.update_suffstats()

    # ---------- Sufficient statistics update ----------
    def update_suffstats(self):
        """Update sufficient statistics for all clusters (including empty ones)."""
        ss = []
        for cluster_id in self.cluster_ids_:
            # collect data assigned to this cluster
            points = np.array([x for x, cid in zip(self.data_, self.assignment) if cid == cluster_id])
            N = points.size
            if N > 0:
                mean = points.mean()
                S = float(np.sum((points - mean) ** 2))
            else:
                # fallback for empty cluster
                mean = float(self.hyperparameters_.get('mean', 0.0))
                S = 0.0
            ss.append(SuffStat(float(mean), int(N), float(S)))
        self.suffstats = ss

    # ---------- Assignment (z) updates ----------
    def log_assignment_score(self, data_id, cluster_id):
        """Compute log-score proportional to log p(z_i = cluster_id | rest)."""
        x = float(self.data_[data_id])
        theta = float(self.cluster_means[cluster_id])
        var = float(self.cluster_variances[cluster_id])
        log_pi = np.log(self.pi[cluster_id])
        # stats.norm.logpdf expects scale = sqrt(variance)
        return log_pi + stats.norm.logpdf(x, loc=theta, scale=np.sqrt(var))

    def assignment_probs(self, data_id):
        """Compute categorical probabilities p(z_i=cid | .) in a numerically stable way."""
        log_scores = np.array([self.log_assignment_score(data_id, cid) for cid in self.cluster_ids_])
        m = log_scores.max()
        weights = np.exp(log_scores - m)
        probs = weights / weights.sum()
        return probs

    def sample_assignment(self, data_id):
        p = self.assignment_probs(data_id)
        return int(self.rng.choice(self.cluster_ids_, p=p))

    def update_assignment(self):
        """Resample cluster assignments for all data points and refresh suffstats."""
        for data_id in range(self.data_.size):
            self.assignment[data_id] = self.sample_assignment(data_id)
        self.update_suffstats()

    # ---------- Mixture weights (pi) updates ----------
    def sample_mixture_weights(self):
        """Sample mixture weights from Dirichlet posterior given counts."""
        ss = self.suffstats
        alpha = [ss[cid].N + self.alpha_ / self.num_clusters_ for cid in self.cluster_ids_]
        return stats.dirichlet(alpha).rvs(size=1).flatten()

    def update_mixture_weights(self):
        self.pi = list(self.sample_mixture_weights())

    # ---------- Cluster parameters: sample tau then mu (Variant B) ----------
    def sample_cluster_params_variance_then_mean(self, cluster_id):
        """Sample variance (tau) then mean (mu) for a single cluster.

        - If N >= 2: tau ~ Inv-Gamma((N-1)/2, S/2), mu | tau ~ Normal(bar, tau/N)
        - If N == 1: use weak proper prior as numerical fallback
        - If N == 0: sample from prior for both tau and mu
        """
        ss = self.suffstats[cluster_id]
        N = ss.N
        bar = ss.theta
        S = ss.S

        # weak proper prior hyperparameters used only as numerical fallback
        alpha0 = 1e-3
        beta0 = 1e-3

        if N >= 2:
            a = (N - 1) / 2.0
            scale = max(S / 2.0, 1e-12)
            tau = float(stats.invgamma(a=a, scale=scale).rvs())
            mu = float(stats.norm(loc=bar, scale=np.sqrt(tau / N)).rvs())
        
        elif N == 1:
            a = alpha0 + 0.5 * N
            scale = beta0 + 0.5 * S
            tau = float(stats.invgamma(a=a, scale=scale).rvs())
            mu = float(stats.norm(loc=bar, scale=np.sqrt(tau / max(1, N))).rvs())
        else:  # N == 0: empty cluster -> sample from prior
            a = alpha0
            scale = beta0
            tau = float(stats.invgamma(a=a, scale=scale).rvs())
            hp_mean = self.hyperparameters_.get('mean', 0.0)
            hp_var = self.hyperparameters_.get('variance', 1e6)
            mu = float(stats.norm(loc=hp_mean, scale=np.sqrt(hp_var)).rvs())
        
        return tau, mu

    def update_cluster_means(self):
        """Update cluster variances and means by sampling tau then mu for each cluster."""
        taus = []
        mus = []
        for cid in self.cluster_ids_:
            tau, mu = self.sample_cluster_params_variance_then_mean(cid)
            taus.append(tau)
            mus.append(mu)
        self.cluster_variances = taus
        self.cluster_means = mus

    # ---------- Log-likelihood, Gibbs step, and trace collection ----------
    def log_likelihood(self):
        """Compute the data log-likelihood log p(X | parameters) under the mixture model.

        This sums log of the mixture density over all data points. Small floor avoids log(0).
        """
        ll = 0.0
        pi = np.asarray(self.pi)
        means = np.asarray(self.cluster_means)
        sds = np.sqrt(np.asarray(self.cluster_variances))
        for x in self.data_:
            pdf_vals = stats.norm.pdf(x, loc=means, scale=sds)
            mix_density = np.dot(pi, pdf_vals)
            mix_density = max(mix_density, 1e-300)
            ll += np.log(mix_density)
        return float(ll)

    def gibbs_step(self):
        """Perform a single Gibbs iteration: resample z, pi, and cluster params, then update suffstats."""
        self.update_assignment()
        self.update_mixture_weights()
        self.update_cluster_means()
        # suffstats are updated by update_assignment; call again to be safe
        self.update_suffstats()

    def sample_trace(self, n_iters, burnin=0, thin=1, collect_ll=False):
        """Run n_iters Gibbs iterations and return samples as an array of shape (d, N).

        The returned vector per iteration concatenates: [means, sigmas, pis] (in this order),
        so d = 3 * K where K is the number of clusters.

        Parameters:
          - n_iters: total number of iterations to run
          - burnin: number of initial iterations to discard
          - thin: keep every 'thin' iteration after burnin
          - collect_ll: if True also return the log-likelihoods for saved iterations

        Returns:
          - samples: ndarray shape (d, N_saved)
          - lls (optional): list of log-likelihoods if collect_ll is True
        """
        K = self.num_clusters_
        d = 3 * K
        saved = []
        lls = []

        total = n_iters
        for it in range(total):
            print(f"Iteration {it}/{total}")
            self.gibbs_step()
            if it >= burnin and ((it - burnin) % thin == 0):
                means = np.asarray(self.cluster_means, dtype=float)
                sigmas = np.sqrt(np.asarray(self.cluster_variances, dtype=float))
                pis = np.asarray(self.pi, dtype=float)
                vec = np.concatenate([means, sigmas, pis])
                saved.append(vec)
                if collect_ll:
                    lls.append(self.log_likelihood())

        if len(saved) == 0:
            return (np.empty((d, 0)), lls) if collect_ll else np.empty((d, 0))

        samples = np.vstack(saved).T  # shape (d, N_saved)
        return (samples, lls) if collect_ll else samples

    # ---------- Utilities / runner ----------
    def run_gibbs(self, n_iters=1, update_pi=True, update_params=True, verbose=False):
        """Run n_iters of Gibbs updates: assignment, mixture weights, and cluster params.

        The boolean flags update_pi and update_params can be used to disable parts for testing.
        """
        for it in range(n_iters):
            self.update_assignment()
            if update_pi:
                self.update_mixture_weights()
            if update_params:
                self.update_cluster_means()
            # suffstats are updated by update_assignment; recompute for safety
            self.update_suffstats()
            if verbose:
                print(f"Iter {it+1}/{n_iters}: counts={Counter(self.assignment)}")

    def get_state(self):
        """Return a dictionary with the main internal state (for inspection)."""
        return {
            'cluster_ids_': self.cluster_ids_,
            'data_': self.data_,
            'num_clusters_': self.num_clusters_,
            'alpha_': self.alpha_,
            'pi': self.pi,
            'assignment': self.assignment,
            'suffstats': self.suffstats,
            'cluster_means': self.cluster_means,
            'cluster_variances': self.cluster_variances,
            'hyperparameters_': self.hyperparameters_
        }

    def summary(self):
        """Return a short textual summary of the current state."""
        counts = Counter(self.assignment)
        s = []
        s.append(f"assignments counts: {dict(counts)}")
        s.append(f"pi: {self.pi}")
        s.append(f"cluster_means: {self.cluster_means}")
        s.append(f"cluster_variances (tau): {self.cluster_variances}")
        return s

    
