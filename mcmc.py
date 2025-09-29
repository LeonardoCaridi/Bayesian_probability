import numpy as np
from scipy.special import logsumexp

"""
def log_prior(theta, bounds):

    for i in range(theta.shape[0]):
        if theta[i] < bounds[i][0] or theta[i] > bounds[i][1]:
            return -np.inf
    
    return 0.0
"""


def log_prior(theta, bounds):
    """
    Prior A:
      - w ~ Beta(2,2)
      - mu1 ~ Normal(0, 3)
      - delta ~ HalfNormal(sd=3)  (delta > 0)
      - sigma1, sigma2 ~ HalfNormal(sd=2)  (sigma > 0)
    bounds = [[w_min,w_max], [mu1_min,mu1_max], [delta_min,delta_max],
              [sigma1_min,sigma1_max], [sigma2_min,sigma2_max]]
    """

    theta = np.atleast_1d(theta)
    try:
        w, mu1, delta, s1, s2 = theta.tolist()
    except Exception:
        # se theta non decomponibile correttamente
        return -np.inf

    # Bounds check (uso gli stessi bounds passati)
    if not (bounds[0][0] <= w <= bounds[0][1]):
        return -np.inf
    if not (bounds[1][0] <= mu1 <= bounds[1][1]):
        return -np.inf
    if not (bounds[2][0] <= delta <= bounds[2][1]):
        return -np.inf
    if not (bounds[3][0] <= s1 <= bounds[3][1]):
        return -np.inf
    if not (bounds[4][0] <= s2 <= bounds[4][1]):
        return -np.inf

    # Additional validity checks (mi assicuro che w sia strettamente tra 0 e 1,
    # e che delta e sigma siano positivi)
    if (w <= 0.0) or (w >= 1.0):
        return -np.inf
    if (delta <= 0.0) or (s1 <= 0.0) or (s2 <= 0.0):
        return -np.inf

    # --- Calcolo log-prior (inclusi termini di normalizzazione) ---
    logp = 0.0

    # Beta(2,2) on w: pdf ∝ w^(1) * (1-w)^(1)
    # normalization Beta(2,2) = 1/6, quindi logpdf = log(w) + log(1-w) - log(1/6)
    # possiamo includere la costante, ma non è necessario per MH: la includo per completezza
    log_beta22_const = -np.log(1.0/6.0)  # = log(6)
    logp += np.log(w) + np.log(1.0 - w) + log_beta22_const

    # mu1 ~ Normal(0, 3)
    sd_mu1 = 3.0
    logp += -0.5 * ((mu1 - 0.0)**2) / (sd_mu1**2) - 0.5 * np.log(2.0 * np.pi * sd_mu1**2)

    # delta ~ HalfNormal(sd=3): pdf(x) = sqrt(2/pi)/sd * exp(-x^2/(2 sd^2))  for x>=0
    sd_delta = 3.0
    if delta < 0:
        return -np.inf
    log_halfnorm_const = 0.5 * (np.log(2.0) - np.log(np.pi)) - np.log(sd_delta)
    logp += log_halfnorm_const - 0.5 * (delta**2) / (sd_delta**2)

    # sigma1, sigma2 ~ HalfNormal(sd=2)
    sd_sigma = 2.0
    log_halfnorm_const_sigma = 0.5 * (np.log(2.0) - np.log(np.pi)) - np.log(sd_sigma)
    logp += log_halfnorm_const_sigma - 0.5 * (s1**2) / (sd_sigma**2)
    logp += log_halfnorm_const_sigma - 0.5 * (s2**2) / (sd_sigma**2)

    return logp
        

def log_T90_distribution(x, theta, sigma_logT90 = 0.0):
    """
    Parametri:
    - x: dati log(T90) (logaritmo naturale)
    - theta: parametri distribuzione
    - sigma_logT90: eventuale errore gaussiano su log(T90)
    """

    w, mu1, delta, sigma1, sigma2 = theta
    if (sigma1 <= 0) or (sigma2 <= 0) or (w <= 0) or (w >= 1) or (delta < 0): return -np.inf
    mu2    = mu1 + delta
    var1   = sigma1**2 + sigma_logT90**2
    var2   = sigma2**2 + sigma_logT90**2

    logN1 = -0.5*(np.log(2*np.pi*var1) + ((x-mu1)**2)/var1)
    logN2 = -0.5*(np.log(2*np.pi*var2) + ((x-mu2)**2)/var2)
    stacked = np.vstack([np.log(w) + logN1, np.log(1-w) + logN2]) 
    # logsumexp(logN1, logN2) = log(N1 + N2), indica la distribuzione di un solo dato
    return logsumexp(stacked, axis=0)     

def log_likelihood(x, theta, sigma_logT90 = 0.0):
    # Assumo i T90 indipendenti e sommo su tutte le probabilità: np.sum()
    return np.sum(log_T90_distribution(x, theta, sigma_logT90), axis=0)

def proposal_distribution(x,  rng = None):
    
    d = x.shape[0]
    #covariance = 0.01*np.eye(d)
    """
    covariance = np.array(([ 1.13628824e-04, 4.76588217e-05, -1.15334653e-05, 4.58064466e-05, -3.70050700e-05],
                           [ 4.76588217e-05, 1.36494158e-03, -1.20039209e-03, 2.62723113e-04, -1.72044907e-04],
                           [-1.15334653e-05, -1.20039209e-03, 1.97755195e-03, -1.15824941e-04, 1.69829670e-05],
                           [ 4.58064466e-05, 2.62723113e-04, -1.15824941e-04, 7.88750561e-04, -1.42154866e-04],
                           [-3.70050700e-05, -1.72044907e-04, 1.69829670e-05,-1.42154866e-04, 5.66782209e-04]))    
    """
    # after adaptive
    covariance = np.array(([ 1.14004786e-04,  5.80630230e-05, -3.52671514e-05,  5.21672529e-05, -3.06139279e-05],
                           [ 5.80630230e-05,  1.26252429e-03, -1.07010900e-03,  2.47320530e-04, -1.83853493e-04],
                           [-3.52671514e-05, -1.07010900e-03,  1.83150662e-03, -9.26694129e-05, -2.11981231e-05],
                           [ 5.21672529e-05,  2.47320530e-04, -9.26694129e-05,  7.82263540e-04, -1.48238595e-04],
                           [-3.06139279e-05, -1.83853493e-04, -2.11981231e-05, -1.48238595e-04, 5.46582734e-04]))
    
    if rng is None:
        rng = np.random
    
    return rng.multivariate_normal(np.zeros(d,dtype=np.float64), covariance)

def log_posterior(x, theta, bounds, sigma_logT90 = 0.0):
    return log_prior(theta, bounds)+log_likelihood(x, theta, sigma_logT90 = sigma_logT90)
    

def generate_data(theta_true, N, rng = None, sigma_logT90 = 1.0):
    """
    Parametri:
    - theta_true: dizionario contenente i parametri noti
    - N: numero sample generati
    - sigma_logT90: rumore gaussiano della misura logT90
    """
    if rng is None:
        rng = np.random

    # Generate synthetic data
    # Genera una maschera: z = 1 seleziona N1 (p = w), z = 0 seleziona N2 (p = 1-w)
    z = rng.random(N) < theta_true["w"] 
    logT90_gen = np.where(z, 
                rng.normal(theta_true["mu1"], theta_true["sigma1"], size=N),
                rng.normal(theta_true["mu2"], theta_true["sigma2"], size=N))

    return logT90_gen + rng.normal(0.0, sigma_logT90, size=N)

def metropolis_hastings(theta0, x, bounds, rng = None, sigma_logT90 = 0.0, n = 1000):
    """
    Parametri: 
    - theta0: initial parameters numpy array
    - x: logT90
    - bounds: intervalli della prior
    - sigma_logT90: eventale errore gaussiano sui dati logT90
    - n: numero di iterazioni
    """
    if rng is None:
        rng = np.random
    
    accepted = 0
    rejected = 1
    
    theta0 = np.atleast_1d(theta0)

    d = theta0.shape[0]
    
    logP0   = log_posterior(x, theta0, bounds, sigma_logT90 = sigma_logT90)
    samples = np.zeros((n,d), dtype=np.float64)
    
    for i in range(n):
        
        theta_t = theta0 + proposal_distribution(theta0, rng = rng)
        logP_t = log_posterior(x, theta_t, bounds, sigma_logT90 = sigma_logT90)
        
#        print('logP proposal = ',logP_t,'logP0 = ',logP0)
        
        if logP_t - logP0 > np.log(rng.uniform(0,1)):
            theta0       = theta_t
            logP0        = logP_t
            samples[i,:] = theta_t
            accepted    += 1
        else:
            samples[i,:] = theta0
            rejected    += 1
        
        print("iteration {0}: acceptance {1}".format(i,accepted/float(accepted+rejected)))
    overall_rate = accepted / float(n)
    print("Adaptive Metropolis finished. Acceptance rate = {:.4f}".format(overall_rate))
    
    return samples

def next_pow_two(n):
    """
    find the next power of 2 given n
    """
    i = 1
    while i < n:
        i = i << 1
    return i

def autocorrelation(x, norm=True):
    """
    compute the autocorrelation function of an array
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf

def init_theta_from_data(x, bounds, rng=None):
    if rng is None:
        rng = np.random()
    # percentili utili
    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)
    mu1_0 = p25
    delta_0 = max(0.5, p75 - p25)   # almeno 0.5
    w0 = 0.5
    s1_0 = 1.0
    s2_0 = 1.0
    # clip to bounds
    theta0 = np.array([np.clip(w0, bounds[0][0]+1e-6, bounds[0][1]-1e-6),
                       np.clip(mu1_0, bounds[1][0], bounds[1][1]),
                       np.clip(delta_0, bounds[2][0]+1e-6, bounds[2][1]),
                       np.clip(s1_0, bounds[3][0], bounds[3][1]),
                       np.clip(s2_0, bounds[4][0], bounds[4][1])], dtype=np.float64)
    return theta0